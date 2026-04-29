import asyncio
import threading
import time
from typing import Optional, Dict, Any, List, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.logger import get_logger
from app.video_source_manager import VideoFrame


logger = get_logger(__name__)


T = TypeVar('T')


class DropStrategy(Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    BLOCK = "block"


@dataclass
class BufferStats:
    max_size: int
    current_size: int
    dropped_frames: int = 0
    total_frames: int = 0
    avg_wait_time_ms: float = 0.0
    last_drop_time: Optional[float] = None


class FrameBuffer(Generic[T]):
    def __init__(
        self,
        max_size: int = 30,
        drop_strategy: DropStrategy = DropStrategy.DROP_OLDEST,
        name: str = "default"
    ):
        self.max_size = max_size
        self.drop_strategy = drop_strategy
        self.name = name
        
        self._buffer: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        
        self._stats = BufferStats(max_size=max_size, current_size=0)
        self._is_closed = False
        
        self._wait_times: List[float] = []
        self._max_wait_samples = 100
        
        logger.info(f"帧缓冲初始化: {name}, 最大容量={max_size}, 丢帧策略={drop_strategy.value}")
    
    def put(self, item: T, timeout: Optional[float] = None) -> bool:
        if self._is_closed:
            return False
        
        with self._lock:
            if self._is_closed:
                return False
            
            while len(self._buffer) >= self.max_size and not self._is_closed:
                if self.drop_strategy == DropStrategy.BLOCK:
                    if not self._not_full.wait(timeout=timeout):
                        return False
                elif self.drop_strategy == DropStrategy.DROP_OLDEST:
                    self._drop_oldest()
                elif self.drop_strategy == DropStrategy.DROP_NEWEST:
                    self._stats.dropped_frames += 1
                    self._stats.last_drop_time = time.time()
                    logger.warning(f"缓冲已满，丢弃最新帧: {self.name}")
                    return False
            
            if self._is_closed:
                return False
            
            self._buffer.append(item)
            self._stats.current_size = len(self._buffer)
            self._stats.total_frames += 1
            
            self._not_empty.notify()
            
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[T]:
        start_time = time.time()
        
        with self._lock:
            while len(self._buffer) == 0 and not self._is_closed:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            
            if self._is_closed and len(self._buffer) == 0:
                return None
            
            item = self._buffer.popleft()
            self._stats.current_size = len(self._buffer)
            
            wait_time = (time.time() - start_time) * 1000
            self._wait_times.append(wait_time)
            if len(self._wait_times) > self._max_wait_samples:
                self._wait_times.pop(0)
            self._stats.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
            
            self._not_full.notify()
            
            return item
    
    def get_batch(self, max_batch_size: int, timeout: Optional[float] = None) -> List[T]:
        items: List[T] = []
        start_time = time.time()
        
        with self._lock:
            while len(self._buffer) == 0 and not self._is_closed:
                elapsed = time.time() - start_time
                if timeout is not None and elapsed >= timeout:
                    return items
                remaining = None if timeout is None else timeout - elapsed
                if not self._not_empty.wait(timeout=remaining):
                    return items
            
            while len(self._buffer) > 0 and len(items) < max_batch_size:
                item = self._buffer.popleft()
                items.append(item)
            
            self._stats.current_size = len(self._buffer)
            self._not_full.notify_all()
            
            return items
    
    def _drop_oldest(self):
        if len(self._buffer) > 0:
            dropped = self._buffer.popleft()
            self._stats.dropped_frames += 1
            self._stats.last_drop_time = time.time()
            self._stats.current_size = len(self._buffer)
            
            if isinstance(dropped, VideoFrame):
                logger.warning(f"缓冲已满，丢弃旧帧: {self.name}, 帧号={dropped.frame_number}")
            else:
                logger.warning(f"缓冲已满，丢弃旧帧: {self.name}")
    
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)
    
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0
    
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self.max_size
    
    def get_stats(self) -> BufferStats:
        with self._lock:
            self._stats.current_size = len(self._buffer)
            return BufferStats(
                max_size=self._stats.max_size,
                current_size=self._stats.current_size,
                dropped_frames=self._stats.dropped_frames,
                total_frames=self._stats.total_frames,
                avg_wait_time_ms=self._stats.avg_wait_time_ms,
                last_drop_time=self._stats.last_drop_time
            )
    
    def clear(self):
        with self._lock:
            old_size = len(self._buffer)
            self._buffer.clear()
            self._stats.current_size = 0
            logger.info(f"清空帧缓冲: {self.name}, 清除了 {old_size} 帧")
            
            self._not_full.notify_all()
    
    def close(self):
        with self._lock:
            self._is_closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()
        
        logger.info(f"帧缓冲已关闭: {self.name}")
    
    def is_closed(self) -> bool:
        with self._lock:
            return self._is_closed


class AsyncFrameBuffer(Generic[T]):
    def __init__(
        self,
        max_size: int = 30,
        drop_strategy: DropStrategy = DropStrategy.DROP_OLDEST,
        name: str = "default"
    ):
        self.max_size = max_size
        self.drop_strategy = drop_strategy
        self.name = name
        
        self._buffer: deque = deque()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._not_full = asyncio.Event()
        self._not_full.set()
        
        self._stats = BufferStats(max_size=max_size, current_size=0)
        self._is_closed = False
        
        self._wait_times: List[float] = []
        self._max_wait_samples = 100
        
        logger.info(f"异步帧缓冲初始化: {name}, 最大容量={max_size}, 丢帧策略={drop_strategy.value}")
    
    async def put(self, item: T, timeout: Optional[float] = None) -> bool:
        if self._is_closed:
            return False
        
        try:
            async with self._lock:
                if self._is_closed:
                    return False
                
                while len(self._buffer) >= self.max_size and not self._is_closed:
                    if self.drop_strategy == DropStrategy.BLOCK:
                        self._not_full.clear()
                        async with self._lock:
                            pass
                        
                        try:
                            await asyncio.wait_for(self._not_full.wait(), timeout=timeout)
                        except asyncio.TimeoutError:
                            return False
                    elif self.drop_strategy == DropStrategy.DROP_OLDEST:
                        self._drop_oldest()
                    elif self.drop_strategy == DropStrategy.DROP_NEWEST:
                        self._stats.dropped_frames += 1
                        self._stats.last_drop_time = time.time()
                        logger.warning(f"缓冲已满，丢弃最新帧: {self.name}")
                        return False
                
                if self._is_closed:
                    return False
                
                self._buffer.append(item)
                self._stats.current_size = len(self._buffer)
                self._stats.total_frames += 1
                
                self._not_empty.set()
                
                return True
                
        except Exception as e:
            logger.error(f"异步帧缓冲put错误: {str(e)}")
            return False
    
    async def get(self, timeout: Optional[float] = None) -> Optional[T]:
        start_time = time.time()
        
        try:
            async with self._lock:
                while len(self._buffer) == 0 and not self._is_closed:
                    self._not_empty.clear()
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return None
                
                if self._is_closed and len(self._buffer) == 0:
                    return None
                
                item = self._buffer.popleft()
                self._stats.current_size = len(self._buffer)
                
                wait_time = (time.time() - start_time) * 1000
                self._wait_times.append(wait_time)
                if len(self._wait_times) > self._max_wait_samples:
                    self._wait_times.pop(0)
                self._stats.avg_wait_time_ms = sum(self._wait_times) / len(self._wait_times)
                
                self._not_full.set()
                
                return item
                
        except Exception as e:
            logger.error(f"异步帧缓冲get错误: {str(e)}")
            return None
    
    async def get_batch(self, max_batch_size: int, timeout: Optional[float] = None) -> List[T]:
        items: List[T] = []
        start_time = time.time()
        
        try:
            async with self._lock:
                while len(self._buffer) == 0 and not self._is_closed:
                    self._not_empty.clear()
                    elapsed = time.time() - start_time
                    remaining = None if timeout is None else timeout - elapsed
                    if remaining is not None and remaining <= 0:
                        return items
                    
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        return items
                
                while len(self._buffer) > 0 and len(items) < max_batch_size:
                    item = self._buffer.popleft()
                    items.append(item)
                
                self._stats.current_size = len(self._buffer)
                self._not_full.set()
                
                return items
                
        except Exception as e:
            logger.error(f"异步帧缓冲get_batch错误: {str(e)}")
            return items
    
    def _drop_oldest(self):
        if len(self._buffer) > 0:
            dropped = self._buffer.popleft()
            self._stats.dropped_frames += 1
            self._stats.last_drop_time = time.time()
            self._stats.current_size = len(self._buffer)
            
            if isinstance(dropped, VideoFrame):
                logger.warning(f"缓冲已满，丢弃旧帧: {self.name}, 帧号={dropped.frame_number}")
            else:
                logger.warning(f"缓冲已满，丢弃旧帧: {self.name}")
    
    async def size(self) -> int:
        async with self._lock:
            return len(self._buffer)
    
    async def is_empty(self) -> bool:
        async with self._lock:
            return len(self._buffer) == 0
    
    async def is_full(self) -> bool:
        async with self._lock:
            return len(self._buffer) >= self.max_size
    
    async def get_stats(self) -> BufferStats:
        async with self._lock:
            self._stats.current_size = len(self._buffer)
            return BufferStats(
                max_size=self._stats.max_size,
                current_size=self._stats.current_size,
                dropped_frames=self._stats.dropped_frames,
                total_frames=self._stats.total_frames,
                avg_wait_time_ms=self._stats.avg_wait_time_ms,
                last_drop_time=self._stats.last_drop_time
            )
    
    async def clear(self):
        async with self._lock:
            old_size = len(self._buffer)
            self._buffer.clear()
            self._stats.current_size = 0
            logger.info(f"清空异步帧缓冲: {self.name}, 清除了 {old_size} 帧")
            
            self._not_full.set()
    
    async def close(self):
        async with self._lock:
            self._is_closed = True
            self._not_empty.set()
            self._not_full.set()
        
        logger.info(f"异步帧缓冲已关闭: {self.name}")
    
    def is_closed(self) -> bool:
        return self._is_closed
