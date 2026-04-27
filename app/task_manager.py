import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import uuid

from app.config import settings
from app.models import (
    TaskBase, TaskCreate, TaskResponse, TaskStatus, 
    TaskType, VideoSourceType, Event, EventType, EventCreate
)
from app.video_detector import VideoDetector
from app.event_manager import EventManager
from app.logger import get_logger


logger = get_logger(__name__)


class TaskManager:
    _instance = None
    _lock = Lock()
    
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
        self._tasks: Dict[str, TaskResponse] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_TASKS)
        self._task_lock = Lock()
        self._initialized = True
        self._load_tasks_from_disk()
    
    def _generate_task_id(self) -> str:
        return f"task_{uuid.uuid4().hex[:12]}"
    
    def _get_task_file_path(self, task_id: str) -> str:
        return os.path.join(settings.TASKS_DIR, f"{task_id}.json")
    
    def _save_task_to_disk(self, task: TaskResponse):
        try:
            task_data = task.model_dump()
            for key, value in task_data.items():
                if isinstance(value, datetime):
                    task_data[key] = value.isoformat()
            with open(self._get_task_file_path(task.id), 'w', encoding='utf-8') as f:
                json.dump(task_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存任务 {task.id} 到磁盘失败: {str(e)}")
    
    def _load_task_from_disk(self, filepath: str) -> Optional[TaskResponse]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            for key, value in task_data.items():
                if key in ['created_at', 'started_at', 'completed_at', 'stopped_at'] and value:
                    task_data[key] = datetime.fromisoformat(value)
            return TaskResponse(**task_data)
        except Exception as e:
            logger.error(f"从磁盘加载任务失败: {str(e)}")
            return None
    
    def _load_tasks_from_disk(self):
        if not os.path.exists(settings.TASKS_DIR):
            return
        for filename in os.listdir(settings.TASKS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(settings.TASKS_DIR, filename)
                task = self._load_task_from_disk(filepath)
                if task:
                    with self._task_lock:
                        self._tasks[task.id] = task
    
    def create_task(self, task_create: TaskCreate) -> TaskResponse:
        task_id = self._generate_task_id()
        
        task = TaskResponse(
            id=task_id,
            name=task_create.name,
            task_type=task_create.task_type,
            video_source=task_create.video_source,
            video_source_type=task_create.video_source_type,
            model_name=task_create.model_name or settings.DEFAULT_MODEL,
            frame_skip=task_create.frame_skip,
            confidence_threshold=task_create.confidence_threshold,
            enable_event_generation=task_create.enable_event_generation,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            progress=0.0,
            current_frame=0,
            events_count=0,
            detections_count=0
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        self._save_task_to_disk(task)
        logger.info(f"创建任务 {task_id}: {task.name}")
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskResponse]:
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TaskResponse]:
        with self._task_lock:
            tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)
    
    async def start_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务 {task_id} 不存在")
            return False
        
        if task.status == TaskStatus.RUNNING:
            logger.warning(f"任务 {task_id} 已在运行中")
            return True
        
        with self._task_lock:
            if task_id in self._running_tasks:
                return True
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self._tasks[task_id] = task
        
        self._save_task_to_disk(task)
        logger.info(f"开始任务 {task_id}: {task.name}")
        
        running_task = asyncio.create_task(self._run_task(task_id))
        self._running_tasks[task_id] = running_task
        return True
    
    async def _run_task(self, task_id: str):
        try:
            task = self.get_task(task_id)
            if not task:
                return
            
            detector = VideoDetector(
                task=task,
                on_progress=self._on_task_progress,
                on_detection=self._on_detection,
                on_event=self._on_event,
                on_error=self._on_task_error
            )
            
            event_manager = EventManager(task_id)
            
            success = await detector.run()
            
            task = self.get_task(task_id)
            if task:
                with self._task_lock:
                    if success:
                        task.status = TaskStatus.COMPLETED
                        task.progress = 100.0
                    else:
                        if task.status != TaskStatus.STOPPED:
                            task.status = TaskStatus.FAILED
                    task.completed_at = datetime.now()
                    self._tasks[task_id] = task
                self._save_task_to_disk(task)
                logger.info(f"任务 {task_id} 完成，状态: {task.status}")
        
        except Exception as e:
            logger.error(f"任务 {task_id} 执行错误: {str(e)}")
            task = self.get_task(task_id)
            if task:
                with self._task_lock:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.completed_at = datetime.now()
                    self._tasks[task_id] = task
                self._save_task_to_disk(task)
        
        finally:
            with self._task_lock:
                self._running_tasks.pop(task_id, None)
    
    def _on_task_progress(self, task_id: str, progress: float, current_frame: int, total_frames: Optional[int], fps: Optional[float]):
        task = self.get_task(task_id)
        if task:
            with self._task_lock:
                task.progress = progress
                task.current_frame = current_frame
                if total_frames:
                    task.total_frames = total_frames
                if fps:
                    task.fps = fps
                self._tasks[task_id] = task
    
    def _on_detection(self, task_id: str, detections_count: int):
        task = self.get_task(task_id)
        if task:
            with self._task_lock:
                task.detections_count += detections_count
                self._tasks[task_id] = task
    
    def _on_event(self, task_id: str, event: Event):
        task = self.get_task(task_id)
        if task:
            with self._task_lock:
                task.events_count += 1
                self._tasks[task_id] = task
    
    def _on_task_error(self, task_id: str, error_message: str):
        task = self.get_task(task_id)
        if task:
            with self._task_lock:
                task.error_message = error_message
                self._tasks[task_id] = task
    
    async def stop_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            logger.error(f"任务 {task_id} 不存在")
            return False
        
        if task.status not in [TaskStatus.RUNNING, TaskStatus.PAUSED]:
            logger.warning(f"任务 {task_id} 不在运行状态")
            return True
        
        with self._task_lock:
            task.status = TaskStatus.STOPPED
            task.stopped_at = datetime.now()
            self._tasks[task_id] = task
        
        self._save_task_to_disk(task)
        
        with self._task_lock:
            running_task = self._running_tasks.get(task_id)
        
        if running_task and not running_task.done():
            running_task.cancel()
            try:
                await running_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"任务 {task_id} 已停止")
        return True
    
    def delete_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False
        
        if task.status == TaskStatus.RUNNING:
            logger.warning(f"任务 {task_id} 正在运行，无法删除")
            return False
        
        with self._task_lock:
            self._tasks.pop(task_id, None)
        
        try:
            filepath = self._get_task_file_path(task_id)
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.error(f"删除任务文件失败: {str(e)}")
        
        logger.info(f"任务 {task_id} 已删除")
        return True


task_manager = TaskManager()
