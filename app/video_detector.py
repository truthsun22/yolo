import asyncio
import cv2
import numpy as np
import torch
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from app.config import settings
from app.models import TaskResponse, TaskType, EventType, Event
from app.event_manager import EventManager
from app.logger import get_logger


logger = get_logger(__name__)


class VideoDetector:
    def __init__(
        self,
        task: TaskResponse,
        on_progress: Optional[Callable] = None,
        on_detection: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        self.task = task
        self.task_id = task.id
        self.on_progress = on_progress
        self.on_detection = on_detection
        self.on_event = on_event
        self.on_error = on_error
        
        self.model_name = task.model_name or settings.DEFAULT_MODEL
        self.frame_skip = task.frame_skip or settings.DEFAULT_FRAME_SKIP
        self.confidence_threshold = task.confidence_threshold or settings.MODEL_CONFIDENCE_THRESHOLD
        self.enable_event_generation = task.enable_event_generation
        
        self.model: Optional[YOLO] = None
        self.event_manager = EventManager(self.task_id)
        self._is_running = False
        self._should_stop = False
        
        logger.info(f"初始化视频检测器: 任务ID={self.task_id}, 模型={self.model_name}")
    
    def _load_model(self):
        try:
            logger.info(f"加载模型: {self.model_name}")
            with torch.serialization.safe_globals([DetectionModel]):
                self.model = YOLO(self.model_name)
            logger.info(f"模型加载完成: {self.model_name}")
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def _get_video_source(self) -> str:
        return self.task.video_source
    
    def _is_people_class(self, class_id: int) -> bool:
        return class_id == 0
    
    def _should_skip_frame(self, frame_count: int) -> bool:
        return frame_count % (self.frame_skip + 1) != 0
    
    async def run(self) -> bool:
        self._is_running = True
        self._should_stop = False
        
        try:
            self._load_model()
            
            video_source = self._get_video_source()
            logger.info(f"打开视频源: {video_source}")
            
            cap = cv2.VideoCapture(video_source)
            
            if not cap.isOpened():
                error_msg = f"无法打开视频源: {video_source}"
                logger.error(error_msg)
                if self.on_error:
                    self.on_error(self.task_id, error_msg)
                return False
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"视频信息: 总帧数={total_frames}, FPS={fps}")
            
            frame_count = 0
            processed_frames = 0
            detections_count = 0
            
            while self._is_running and not self._should_stop:
                ret, frame = cap.read()
                
                if not ret:
                    logger.info("视频读取完成")
                    break
                
                frame_count += 1
                
                if self._should_skip_frame(frame_count):
                    continue
                
                processed_frames += 1
                
                results = await self._process_frame(frame, frame_count)
                
                if results and len(results) > 0:
                    frame_detections = self._count_people_detections(results)
                    detections_count += frame_detections
                    
                    if self.on_detection:
                        self.on_detection(self.task_id, frame_detections)
                    
                    if self.enable_event_generation:
                        await self._generate_events(results, frame_count)
                
                if total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                else:
                    progress = (processed_frames % 100)
                
                if self.on_progress:
                    self.on_progress(
                        self.task_id,
                        min(progress, 100.0),
                        frame_count,
                        total_frames if total_frames > 0 else None,
                        fps if fps > 0 else None
                    )
                
                await asyncio.sleep(0)
            
            cap.release()
            self._is_running = False
            
            logger.info(f"视频处理完成: 总帧数={frame_count}, 处理帧数={processed_frames}, 检测数量={detections_count}")
            
            EventManager.clear_task_cache(self.task_id)
            
            return True
            
        except asyncio.CancelledError:
            logger.info(f"任务 {self.task_id} 被取消")
            self._is_running = False
            return False
        except Exception as e:
            error_msg = f"视频处理错误: {str(e)}"
            logger.error(error_msg)
            if self.on_error:
                self.on_error(self.task_id, error_msg)
            self._is_running = False
            return False
    
    async def _process_frame(self, frame: np.ndarray, frame_count: int) -> list:
        try:
            if self.model is None:
                return []
            
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=settings.MODEL_IOU_THRESHOLD,
                verbose=False
            )
            
            return results
            
        except Exception as e:
            logger.error(f"处理帧 {frame_count} 时出错: {str(e)}")
            return []
    
    def _count_people_detections(self, results) -> int:
        count = 0
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    if self._is_people_class(class_id):
                        count += 1
        return count
    
    async def _generate_events(self, results, frame_number: int):
        try:
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for i, box in enumerate(boxes):
                    class_id = int(box.cls[0])
                    
                    if not self._is_people_class(class_id):
                        continue
                    
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bounding_box = {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": x2 - x1,
                        "height": y2 - y1
                    }
                    
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0])
                    
                    event_type = self._get_event_type()
                    
                    event = self.event_manager.create_event(
                        event_type=event_type,
                        frame_number=frame_number,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        track_id=track_id,
                        metadata={
                            "task_type": self.task.task_type.value,
                            "model_name": self.model_name
                        }
                    )
                    
                    if event and self.on_event:
                        self.on_event(self.task_id, event)
                        logger.debug(
                            f"生成事件: 任务={self.task_id}, "
                            f"帧={frame_number}, "
                            f"置信度={confidence:.2f}, "
                            f"追踪ID={track_id}"
                        )
                        
        except Exception as e:
            logger.error(f"生成事件时出错: {str(e)}")
    
    def _get_event_type(self) -> EventType:
        if self.task.task_type == TaskType.PEDESTRIAN_TRACKING:
            return EventType.PEDESTRIAN_TRACKED
        return EventType.PEDESTRIAN_DETECTED
    
    def stop(self):
        self._should_stop = True
        self._is_running = False
        logger.info(f"停止视频检测器: 任务ID={self.task_id}")
