import asyncio
import cv2
import numpy as np
import os
import torch
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, Any, Callable, List, Tuple
from ultralytics import YOLO

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
        
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.output_video_path: str = ""
        self.output_fps: float = 30.0
        self.output_frame_size: Optional[Tuple[int, int]] = None
        self._used_codec: str = ""
        
        self.screenshot_count = 0
        self.event_screenshots: List[Dict[str, Any]] = []
        
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        
        logger.info(f"初始化视频检测器: 任务ID={self.task_id}, 模型={self.model_name}")
    
    def _load_model(self):
        try:
            logger.info(f"加载模型: {self.model_name}")
            
            original_torch_load = torch.load
            
            @wraps(original_torch_load)
            def patched_torch_load(*args, **kwargs):
                if 'weights_only' not in kwargs or kwargs['weights_only']:
                    kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_torch_load
            
            try:
                self.model = YOLO(self.model_name)
            finally:
                torch.load = original_torch_load
            
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
    
    def _get_output_video_path(self, codec_fourcc: str) -> str:
        extension = '.mp4'
        if codec_fourcc == 'MJPG':
            extension = '.avi'
        elif codec_fourcc == 'XVID':
            extension = '.avi'
        return os.path.join(settings.VIDEO_OUTPUT_DIR, f"{self.task_id}_annotated{extension}")
    
    def _get_screenshot_path(self, frame_number: int, track_id: Optional[int] = None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if track_id is not None:
            filename = f"{self.task_id}_frame{frame_number}_track{track_id}_{timestamp}.jpg"
        else:
            filename = f"{self.task_id}_frame{frame_number}_{timestamp}.jpg"
        return os.path.join(settings.SCREENSHOTS_DIR, filename)
    
    def _init_video_writer(self, frame_width: int, frame_height: int, fps: float):
        try:
            if self.video_writer is not None:
                self.video_writer.release()
            
            self.output_frame_size = (frame_width, frame_height)
            self.output_fps = fps if fps > 0 else 30.0
            
            codecs_to_try = [
                ('avc1', 'H.264 (avc1)'),
                ('H264', 'H.264'),
                ('mp4v', 'MPEG-4'),
                ('MJPG', 'Motion JPEG'),
            ]
            
            for codec_fourcc, codec_name in codecs_to_try:
                try:
                    self.output_video_path = self._get_output_video_path(codec_fourcc)
                    
                    fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
                    self.video_writer = cv2.VideoWriter(
                        self.output_video_path,
                        fourcc,
                        self.output_fps,
                        self.output_frame_size
                    )
                    
                    if self.video_writer and self.video_writer.isOpened():
                        self._used_codec = codec_fourcc
                        logger.info(f"视频写入器初始化成功: {self.output_video_path}, 编码={codec_name}, 分辨率={frame_width}x{frame_height}, FPS={self.output_fps}")
                        return
                    else:
                        logger.warning(f"编码 {codec_name} 初始化失败，尝试下一个编码")
                        if self.video_writer:
                            self.video_writer.release()
                            self.video_writer = None
                except Exception as codec_error:
                    logger.warning(f"编码 {codec_name} 初始化异常: {str(codec_error)}，尝试下一个编码")
            
            logger.error(f"所有可用编码都无法初始化视频写入器")
            self.video_writer = None
            
        except Exception as e:
            logger.error(f"初始化视频写入器失败: {str(e)}")
            self.video_writer = None
    
    def _get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"class_{class_id}")
    
    def _draw_detections(self, frame: np.ndarray, results, frame_number: int) -> np.ndarray:
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if confidence < self.confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0])
                
                color = self._get_color_for_class(class_id, track_id)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                class_name = self._get_class_name(class_id)
                if track_id is not None:
                    label = f"{class_name} #{track_id} ({confidence:.2f})"
                else:
                    label = f"{class_name} ({confidence:.2f})"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 10, y1),
                    color,
                    -1
                )
                
                text_color = (255, 255, 255) if self._is_dark_color(color) else (0, 0, 0)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 5, y1 - 5),
                    font,
                    font_scale,
                    text_color,
                    font_thickness
                )
        
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        task_type_text = "Tracking" if self.task.task_type == TaskType.PEDESTRIAN_TRACKING else "Detection"
        cv2.putText(
            annotated_frame,
            f"Task Type: {task_type_text}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        return annotated_frame
    
    def _get_color_for_class(self, class_id: int, track_id: Optional[int] = None) -> Tuple[int, int, int]:
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (0, 128, 255), (255, 192, 203), (165, 42, 42), (0, 100, 0), (75, 0, 130)
        ]
        
        if track_id is not None:
            return colors[track_id % len(colors)]
        
        return colors[class_id % len(colors)]
    
    def _is_dark_color(self, color: Tuple[int, int, int]) -> bool:
        brightness = (color[2] * 299 + color[1] * 587 + color[0] * 114) / 1000
        return brightness < 128
    
    def _save_event_screenshot(self, frame: np.ndarray, event: Event) -> Optional[str]:
        try:
            screenshot_path = self._get_screenshot_path(event.frame_number, event.track_id)
            
            if cv2.imwrite(screenshot_path, frame):
                self.screenshot_count += 1
                self.event_screenshots.append({
                    "event_id": event.id,
                    "frame_number": event.frame_number,
                    "track_id": event.track_id,
                    "screenshot_path": screenshot_path,
                    "timestamp": datetime.now().isoformat()
                })
                logger.info(f"事件截图已保存: {screenshot_path}")
                return screenshot_path
            else:
                logger.error(f"保存事件截图失败: {screenshot_path}")
                return None
        except Exception as e:
            logger.error(f"保存事件截图时出错: {str(e)}")
            return None
    
    def _close_video_writer(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.info(f"视频写入器已关闭，输出文件: {self.output_video_path}")
    
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
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: 总帧数={total_frames}, FPS={fps}, 分辨率={frame_width}x{frame_height}")
            
            self._init_video_writer(frame_width, frame_height, fps)
            
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
                
                annotated_frame = self._draw_detections(frame, results, frame_count)
                
                if self.video_writer is not None:
                    self.video_writer.write(annotated_frame)
                
                if results and len(results) > 0:
                    frame_detections = self._count_people_detections(results)
                    detections_count += frame_detections
                    
                    if self.on_detection:
                        self.on_detection(self.task_id, frame_detections)
                    
                    if self.enable_event_generation:
                        await self._generate_events(results, frame_count, annotated_frame)
                
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
            self._close_video_writer()
            self._is_running = False
            
            logger.info(f"视频处理完成: 总帧数={frame_count}, 处理帧数={processed_frames}, 检测数量={detections_count}, 截图数量={self.screenshot_count}")
            
            EventManager.clear_task_cache(self.task_id)
            
            return True
            
        except asyncio.CancelledError:
            logger.info(f"任务 {self.task_id} 被取消")
            self._close_video_writer()
            self._is_running = False
            return False
        except Exception as e:
            error_msg = f"视频处理错误: {str(e)}"
            logger.error(error_msg)
            self._close_video_writer()
            if self.on_error:
                self.on_error(self.task_id, error_msg)
            self._is_running = False
            return False
    
    async def _process_frame(self, frame: np.ndarray, frame_count: int) -> list:
        try:
            if self.model is None:
                return []
            
            if self.task.task_type == TaskType.PEDESTRIAN_TRACKING:
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=settings.MODEL_IOU_THRESHOLD,
                    verbose=False,
                    persist=True
                )
            else:
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
    
    async def _generate_events(self, results, frame_number: int, annotated_frame: Optional[np.ndarray] = None):
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
                        
                        if annotated_frame is not None:
                            screenshot_path = self._save_event_screenshot(annotated_frame, event)
                            if screenshot_path:
                                logger.info(f"事件截图已保存: {screenshot_path}")
                        
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
