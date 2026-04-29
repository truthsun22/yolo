import asyncio
import torch
import threading
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from ultralytics import YOLO
from dataclasses import dataclass
from enum import Enum

from app.config import settings
from app.logger import get_logger


logger = get_logger(__name__)


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class DeviceInfo:
    device_type: DeviceType
    device_index: int = 0
    name: str = ""
    memory_gb: float = 0.0
    is_available: bool = True


class ModelHandle:
    def __init__(self, model: YOLO, model_name: str, device: str):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.reference_count: int = 0
        self._lock = threading.Lock()
    
    def acquire(self):
        with self._lock:
            self.reference_count += 1
    
    def release(self):
        with self._lock:
            self.reference_count -= 1
            return self.reference_count <= 0


class DeviceManager:
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
        
        self._models: Dict[str, ModelHandle] = {}
        self._model_lock = threading.Lock()
        
        self._available_devices: List[DeviceInfo] = []
        self._preferred_device: Optional[DeviceInfo] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        
        self._detect_devices()
        self._initialized = True
        
        logger.info(f"设备管理器初始化完成，首选设备: {self._preferred_device.device_type.value if self._preferred_device else 'None'}")
    
    def _detect_devices(self):
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            for i in range(cuda_device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                
                self._available_devices.append(DeviceInfo(
                    device_type=DeviceType.CUDA,
                    device_index=i,
                    name=device_name,
                    memory_gb=device_memory,
                    is_available=True
                ))
                logger.info(f"检测到CUDA设备 {i}: {device_name}, 显存: {device_memory:.2f} GB")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self._available_devices.append(DeviceInfo(
                device_type=DeviceType.MPS,
                device_index=0,
                name="Apple Silicon MPS",
                memory_gb=0.0,
                is_available=True
            ))
            logger.info("检测到MPS设备")
        
        self._available_devices.append(DeviceInfo(
            device_type=DeviceType.CPU,
            device_index=0,
            name="CPU",
            memory_gb=0.0,
            is_available=True
        ))
        
        if self._available_devices:
            for device in self._available_devices:
                if device.device_type != DeviceType.CPU:
                    self._preferred_device = device
                    break
            
            if not self._preferred_device:
                self._preferred_device = self._available_devices[-1]
    
    def get_device_string(self, force_cpu: bool = False) -> str:
        if force_cpu or not self._preferred_device:
            return "cpu"
        
        if self._preferred_device.device_type == DeviceType.CUDA:
            return f"cuda:{self._preferred_device.device_index}"
        elif self._preferred_device.device_type == DeviceType.MPS:
            return "mps"
        else:
            return "cpu"
    
    def get_available_devices(self) -> List[DeviceInfo]:
        return list(self._available_devices)
    
    def is_gpu_available(self) -> bool:
        return any(d.device_type != DeviceType.CPU and d.is_available for d in self._available_devices)
    
    def get_model(self, model_name: str, force_cpu: bool = False) -> ModelHandle:
        device = self.get_device_string(force_cpu=force_cpu)
        cache_key = f"{model_name}_{device}"
        
        with self._model_lock:
            if cache_key in self._models:
                handle = self._models[cache_key]
                handle.acquire()
                logger.debug(f"复用已加载的模型: {model_name} on {device}, 引用计数: {handle.reference_count}")
                return handle
        
        model = self._load_model(model_name, device)
        
        with self._model_lock:
            if cache_key in self._models:
                del model
                handle = self._models[cache_key]
                handle.acquire()
                return handle
            
            handle = ModelHandle(model, model_name, device)
            handle.acquire()
            self._models[cache_key] = handle
            logger.info(f"模型加载成功: {model_name} on {device}")
            return handle
    
    def _load_model(self, model_name: str, device: str) -> YOLO:
        try:
            logger.info(f"加载模型: {model_name} 到设备: {device}")
            
            original_torch_load = torch.load
            
            @wraps(original_torch_load)
            def patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs or kwargs['weights_only']:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_torch_load
            
            try:
                model = YOLO(model_name)
                model.to(device)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                logger.info(f"模型加载完成: {model_name} on {device}")
                return model
            finally:
                torch.load = original_torch_load
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def release_model(self, handle: ModelHandle):
        if handle.release():
            cache_key = f"{handle.model_name}_{handle.device}"
            with self._model_lock:
                if cache_key in self._models and self._models[cache_key] is handle:
                    del self._models[cache_key]
                    
                    if handle.device.startswith('cuda'):
                        if hasattr(handle.model, 'model'):
                            del handle.model.model
                        del handle.model
                        torch.cuda.empty_cache()
                    
                    logger.info(f"模型已卸载: {handle.model_name} on {handle.device}")
    
    async def run_inference(
        self,
        model_handle: ModelHandle,
        frames,
        task_type: str = "detect",
        conf: float = 0.5,
        iou: float = 0.5,
        persist: bool = False
    ):
        model = model_handle.model
        device = model_handle.device
        
        if task_type == "track":
            results = model.track(
                frames,
                conf=conf,
                iou=iou,
                verbose=False,
                persist=persist
            )
        else:
            results = model(
                frames,
                conf=conf,
                iou=iou,
                verbose=False
            )
        
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        
        return results
    
    def get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            max_workers = settings.MAX_CONCURRENT_TASKS if hasattr(settings, 'MAX_CONCURRENT_TASKS') else 5
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
        return self._executor
    
    def shutdown(self):
        with self._model_lock:
            for cache_key in list(self._models.keys()):
                handle = self._models[cache_key]
                if handle.device.startswith('cuda'):
                    if hasattr(handle.model, 'model'):
                        del handle.model.model
                    del handle.model
                del self._models[cache_key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None
        
        logger.info("设备管理器已关闭")


device_manager = DeviceManager()
