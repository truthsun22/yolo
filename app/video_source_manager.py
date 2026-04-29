import asyncio
import cv2
import threading
import time
from typing import Optional, Dict, Any, List, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty, Full

from app.config import settings
from app.logger import get_logger


logger = get_logger(__name__)


class VideoSourceType(Enum):
    FILE = "file"
    RTSP = "rtsp"
    WEBCAM = "webcam"
    UNKNOWN = "unknown"


@dataclass
class VideoFrame:
    frame: Any
    frame_number: int
    timestamp: float
    source_id: str


@dataclass
class VideoSourceInfo:
    source_path: str
    source_type: VideoSourceType
    fps: float = 0.0
    width: int = 0
    height: int = 0
    total_frames: int = 0
    is_live: bool = False


class VideoSourceReader:
    def __init__(self, source_path: str, source_id: str):
        self.source_path = source_path
        self.source_id = source_id
        self.source_info: Optional[VideoSourceInfo] = None
        
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._should_stop = False
        self._read_thread: Optional[threading.Thread] = None
        
        self._listeners: Dict[str, Callable[[VideoFrame], None]] = {}
        self._listener_lock = threading.Lock()
        
        self._frame_count = 0
        self._start_time = 0.0
        
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 2.0
    
    def _detect_source_type(self) -> VideoSourceType:
        if self.source_path.startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            return VideoSourceType.RTSP
        try:
            int(self.source_path)
            return VideoSourceType.WEBCAM
        except ValueError:
            return VideoSourceType.FILE
    
    def _open_capture(self) -> bool:
        try:
            if self._cap is not None:
                self._cap.release()
            
            source = self.source_path
            if self._detect_source_type() == VideoSourceType.WEBCAM:
                source = int(self.source_path)
            
            self._cap = cv2.VideoCapture(source)
            
            if self._detect_source_type() == VideoSourceType.RTSP:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self._cap.isOpened():
                logger.error(f"无法打开视频源: {self.source_path}")
                return False
            
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            source_type = self._detect_source_type()
            is_live = source_type in [VideoSourceType.RTSP, VideoSourceType.WEBCAM]
            
            self.source_info = VideoSourceInfo(
                source_path=self.source_path,
                source_type=source_type,
                fps=fps if fps > 0 else 30.0,
                width=width,
                height=height,
                total_frames=total_frames,
                is_live=is_live
            )
            
            logger.info(f"视频源已打开: {self.source_path}, 类型={source_type.value}, "
                       f"分辨率={width}x{height}, FPS={self.source_info.fps}")
            return True
            
        except Exception as e:
            logger.error(f"打开视频源时出错: {str(e)}")
            return False
    
    def add_listener(self, task_id: str, callback: Callable[[VideoFrame], None]):
        with self._listener_lock:
            self._listeners[task_id] = callback
            logger.debug(f"添加监听器: 任务={task_id}, 视频源={self.source_id}")
    
    def remove_listener(self, task_id: str):
        with self._listener_lock:
            self._listeners.pop(task_id, None)
            logger.debug(f"移除监听器: 任务={task_id}, 视频源={self.source_id}")
    
    def has_listeners(self) -> bool:
        with self._listener_lock:
            return len(self._listeners) > 0
    
    def start(self):
        if self._is_running:
            return
        
        if not self._open_capture():
            raise RuntimeError(f"无法打开视频源: {self.source_path}")
        
        self._is_running = True
        self._should_stop = False
        self._frame_count = 0
        self._start_time = time.time()
        self._reconnect_attempts = 0
        
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        
        logger.info(f"视频源读取器已启动: {self.source_id}")
    
    def stop(self):
        self._should_stop = True
        
        if self._read_thread and self._read_thread.is_alive():
            self._read_thread.join(timeout=5.0)
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        self._is_running = False
        logger.info(f"视频源读取器已停止: {self.source_id}")
    
    def _read_loop(self):
        while self._is_running and not self._should_stop:
            try:
                if self._cap is None or not self._cap.isOpened():
                    if not self._handle_reconnect():
                        break
                    continue
                
                ret, frame = self._cap.read()
                
                if not ret:
                    if self.source_info and self.source_info.is_live:
                        logger.warning(f"视频源读取失败，尝试重连: {self.source_id}")
                        if not self._handle_reconnect():
                            break
                        continue
                    else:
                        logger.info(f"视频源读取完成: {self.source_id}")
                        break
                
                self._frame_count += 1
                
                video_frame = VideoFrame(
                    frame=frame,
                    frame_number=self._frame_count,
                    timestamp=time.time(),
                    source_id=self.source_id
                )
                
                self._notify_listeners(video_frame)
                
                if self.source_info and not self.source_info.is_live:
                    target_delay = 1.0 / self.source_info.fps
                    elapsed = time.time() - video_frame.timestamp
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)
                    
            except Exception as e:
                logger.error(f"视频读取循环错误: {str(e)}")
                if self.source_info and self.source_info.is_live:
                    if not self._handle_reconnect():
                        break
                else:
                    break
    
    def _handle_reconnect(self) -> bool:
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"重连尝试次数已达上限: {self.source_id}")
            return False
        
        self._reconnect_attempts += 1
        logger.info(f"尝试重连视频源 ({self._reconnect_attempts}/{self._max_reconnect_attempts}): {self.source_id}")
        
        time.sleep(self._reconnect_delay)
        
        try:
            if self._open_capture():
                self._reconnect_attempts = 0
                return True
        except Exception as e:
            logger.error(f"重连失败: {str(e)}")
        
        return False
    
    def _notify_listeners(self, frame: VideoFrame):
        with self._listener_lock:
            listeners = list(self._listeners.values())
        
        for callback in listeners:
            try:
                callback(frame)
            except Exception as e:
                logger.error(f"通知监听器失败: {str(e)}")


class VideoSourceManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._readers: Dict[str, VideoSourceReader] = {}
        self._reader_lock = threading.Lock()
        self._task_to_source: Dict[str, str] = {}
        
        self._initialized = True
        logger.info("视频源管理器初始化完成")
    
    def _get_source_id(self, source_path: str) -> str:
        return f"source_{hash(source_path) & 0xFFFFFFFF:08x}"
    
    def get_or_create_reader(self, source_path: str) -> VideoSourceReader:
        source_id = self._get_source_id(source_path)
        
        with self._reader_lock:
            if source_id in self._readers:
                return self._readers[source_id]
            
            reader = VideoSourceReader(source_path, source_id)
            self._readers[source_id] = reader
            
            logger.info(f"创建视频源读取器: {source_id} -> {source_path}")
            return reader
    
    def subscribe(self, task_id: str, source_path: str, callback: Callable[[VideoFrame], None]) -> VideoSourceReader:
        source_id = self._get_source_id(source_path)
        
        reader = self.get_or_create_reader(source_path)
        
        reader.add_listener(task_id, callback)
        
        with self._reader_lock:
            self._task_to_source[task_id] = source_id
        
        if not reader._is_running:
            reader.start()
        
        return reader
    
    def unsubscribe(self, task_id: str):
        with self._reader_lock:
            source_id = self._task_to_source.pop(task_id, None)
        
        if source_id:
            with self._reader_lock:
                reader = self._readers.get(source_id)
            
            if reader:
                reader.remove_listener(task_id)
                
                if not reader.has_listeners():
                    reader.stop()
                    with self._reader_lock:
                        if source_id in self._readers:
                            del self._readers[source_id]
                    logger.info(f"视频源读取器已销毁: {source_id}")
    
    def get_reader_info(self, source_path: str) -> Optional[VideoSourceInfo]:
        source_id = self._get_source_id(source_path)
        
        with self._reader_lock:
            reader = self._readers.get(source_id)
        
        if reader and reader.source_info:
            return reader.source_info
        return None
    
    def stop_all(self):
        with self._reader_lock:
            readers = list(self._readers.values())
        
        for reader in readers:
            reader.stop()
        
        with self._reader_lock:
            self._readers.clear()
            self._task_to_source.clear()
        
        logger.info("所有视频源读取器已停止")


video_source_manager = VideoSourceManager()
