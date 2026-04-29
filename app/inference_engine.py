import asyncio
import threading
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
from collections import defaultdict

from app.config import settings
from app.logger import get_logger
from app.device_manager import device_manager, DeviceManager, ModelHandle
from app.frame_buffer import FrameBuffer, AsyncFrameBuffer, DropStrategy


logger = get_logger(__name__)


class InferenceMode(Enum):
    DETECT = "detect"
    TRACK = "track"


@dataclass
class InferenceRequest:
    request_id: str
    task_id: str
    frames: List[np.ndarray]
    frame_numbers: List[int]
    mode: InferenceMode
    conf_threshold: float
    iou_threshold: float
    persist: bool = False
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class InferenceResult:
    request_id: str
    task_id: str
    frame_numbers: List[int]
    results: List[Any]
    inference_time_ms: float
    success: bool = True
    error_message: Optional[str] = None


class InferenceEngine:
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
        
        self._batch_size = getattr(settings, 'INFERENCE_BATCH_SIZE', 4)
        self._max_queue_size = getattr(settings, 'INFERENCE_MAX_QUEUE', 100)
        self._batch_timeout_ms = getattr(settings, 'INFERENCE_BATCH_TIMEOUT', 100)
        
        self._request_queue: Queue = Queue(maxsize=self._max_queue_size)
        self._results: Dict[str, InferenceResult] = {}
        self._result_callbacks: Dict[str, Callable] = {}
        
        self._model_handles: Dict[str, ModelHandle] = {}
        
        self._is_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        self._stats = {
            'total_requests': 0,
            'total_batches': 0,
            'total_frames_processed': 0,
            'avg_inference_time_ms': 0.0,
            'dropped_requests': 0
        }
        self._stats_lock = threading.Lock()
        
        self._initialized = True
        logger.info(f"推理引擎初始化完成: 批处理大小={self._batch_size}, "
                   f"最大队列大小={self._max_queue_size}, 批处理超时={self._batch_timeout_ms}ms")
    
    def start(self):
        if self._is_running:
            return
        
        self._is_running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("推理引擎已启动")
    
    def stop(self):
        self._is_running = False
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        
        for model_name, handle in list(self._model_handles.items()):
            device_manager.release_model(handle)
        self._model_handles.clear()
        
        logger.info("推理引擎已停止")
    
    def _get_model_handle(self, model_name: str, force_cpu: bool = False) -> ModelHandle:
        cache_key = f"{model_name}_{'cpu' if force_cpu else 'gpu'}"
        
        if cache_key in self._model_handles:
            return self._model_handles[cache_key]
        
        handle = device_manager.get_model(model_name, force_cpu=force_cpu)
        self._model_handles[cache_key] = handle
        
        return handle
    
    def submit(
        self,
        task_id: str,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        model_name: str,
        mode: InferenceMode = InferenceMode.DETECT,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        persist: bool = False,
        callback: Optional[Callable] = None,
        force_cpu: bool = False
    ) -> Optional[str]:
        if not self._is_running:
            logger.warning("推理引擎未运行，无法提交请求")
            return None
        
        request_id = f"req_{int(time.time() * 1000)}_{id(frames)}"
        
        request = InferenceRequest(
            request_id=request_id,
            task_id=task_id,
            frames=frames,
            frame_numbers=frame_numbers,
            mode=mode,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            persist=persist,
            callback=callback
        )
        
        try:
            self._request_queue.put(request, block=False)
            
            if callback:
                self._result_callbacks[request_id] = callback
            
            logger.debug(f"已提交推理请求: {request_id}, 任务={task_id}, 帧数={len(frames)}")
            return request_id
            
        except:
            logger.warning(f"推理队列已满，丢弃请求: {request_id}")
            with self._stats_lock:
                self._stats['dropped_requests'] += 1
            return None
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[InferenceResult]:
        start_time = time.time()
        
        while True:
            if request_id in self._results:
                result = self._results.pop(request_id)
                return result
            
            elapsed = time.time() - start_time
            if timeout is not None and elapsed >= timeout:
                return None
            
            time.sleep(0.01)
    
    def _worker_loop(self):
        logger.info("推理引擎工作线程已启动")
        
        while self._is_running:
            try:
                batch: List[InferenceRequest] = []
                deadline = time.time() + (self._batch_timeout_ms / 1000.0)
                
                while len(batch) < self._batch_size and time.time() < deadline:
                    try:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        
                        request = self._request_queue.get(timeout=min(remaining, 0.1))
                        batch.append(request)
                        
                    except Empty:
                        break
                
                if batch:
                    self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"推理引擎工作线程错误: {str(e)}")
                time.sleep(0.1)
        
        logger.info("推理引擎工作线程已停止")
    
    def _process_batch(self, batch: List[InferenceRequest]):
        if not batch:
            return
        
        start_time = time.time()
        
        model_name = batch[0].task_id
        all_frames: List[np.ndarray] = []
        all_frame_numbers: List[Tuple[int, int]] = []
        
        for request_idx, request in enumerate(batch):
            for frame_idx, frame in enumerate(request.frames):
                all_frames.append(frame)
                all_frame_numbers.append((request_idx, request.frame_numbers[frame_idx]))
        
        try:
            handle = self._get_model_handle(batch[0].task_id)
            
            inference_start = time.time()
            
            if len(all_frames) == 1:
                results = handle.model(
                    all_frames[0],
                    conf=batch[0].conf_threshold,
                    iou=batch[0].iou_threshold,
                    verbose=False
                )
                results = [results]
            else:
                results = handle.model(
                    all_frames,
                    conf=batch[0].conf_threshold,
                    iou=batch[0].iou_threshold,
                    verbose=False
                )
            
            inference_time_ms = (time.time() - inference_start) * 1000
            
            request_results: Dict[int, List[Any]] = defaultdict(list)
            request_frame_numbers: Dict[int, List[int]] = defaultdict(list)
            
            for i, ((request_idx, frame_number), result) in enumerate(zip(all_frame_numbers, results)):
                request_results[request_idx].append(result)
                request_frame_numbers[request_idx].append(frame_number)
            
            for request_idx, request in enumerate(batch):
                result = InferenceResult(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    frame_numbers=request_frame_numbers.get(request_idx, []),
                    results=request_results.get(request_idx, []),
                    inference_time_ms=inference_time_ms / len(batch)
                )
                
                self._results[request.request_id] = result
                
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as e:
                        logger.error(f"执行推理回调错误: {str(e)}")
                
                self._result_callbacks.pop(request.request_id, None)
            
            with self._stats_lock:
                self._stats['total_requests'] += len(batch)
                self._stats['total_batches'] += 1
                self._stats['total_frames_processed'] += len(all_frames)
                self._stats['avg_inference_time_ms'] = (
                    (self._stats['avg_inference_time_ms'] * (self._stats['total_batches'] - 1) + inference_time_ms)
                    / self._stats['total_batches']
                )
            
            logger.debug(f"处理推理批: 请求数={len(batch)}, 帧数={len(all_frames)}, "
                        f"推理时间={inference_time_ms:.2f}ms, 单帧={inference_time_ms/len(all_frames):.2f}ms")
            
        except Exception as e:
            logger.error(f"推理批处理错误: {str(e)}")
            
            for request in batch:
                result = InferenceResult(
                    request_id=request.request_id,
                    task_id=request.task_id,
                    frame_numbers=request.frame_numbers,
                    results=[],
                    inference_time_ms=0,
                    success=False,
                    error_message=str(e)
                )
                
                self._results[request.request_id] = result
                
                if request.callback:
                    try:
                        request.callback(result)
                    except Exception as cb_e:
                        logger.error(f"执行错误回调错误: {str(cb_e)}")
                
                self._result_callbacks.pop(request.request_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)
    
    def get_queue_size(self) -> int:
        return self._request_queue.qsize()


class AsyncInferenceEngine:
    def __init__(self):
        self._engine = InferenceEngine()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._engine.start()
        logger.info("异步推理引擎已启动")
    
    async def stop(self):
        self._engine.stop()
        self._executor.shutdown(wait=False)
        logger.info("异步推理引擎已停止")
    
    async def infer(
        self,
        task_id: str,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        model_name: str,
        mode: InferenceMode = InferenceMode.DETECT,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        persist: bool = False,
        force_cpu: bool = False
    ) -> Optional[InferenceResult]:
        result_future: asyncio.Future = asyncio.Future()
        
        def callback(result: InferenceResult):
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(result_future.set_result, result)
        
        request_id = self._engine.submit(
            task_id=task_id,
            frames=frames,
            frame_numbers=frame_numbers,
            model_name=model_name,
            mode=mode,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            persist=persist,
            callback=callback,
            force_cpu=force_cpu
        )
        
        if request_id is None:
            return None
        
        try:
            result = await asyncio.wait_for(result_future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            logger.error(f"推理请求超时: {request_id}")
            return None


inference_engine = InferenceEngine()
