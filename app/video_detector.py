import asyncio
import cv2
import numpy as np
import os
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.models import TaskResponse, TaskType, EventType, Event
from app.event_manager import EventManager
from app.logger import get_logger
from app.device_manager import device_manager, ModelHandle
from app.video_source_manager import video_source_manager, VideoFrame, VideoSourceInfo
from app.frame_buffer import AsyncFrameBuffer, DropStrategy, BufferStats
from app.inference_engine import InferenceMode


logger = get_logger(__name__)


@dataclass
class ProcessedFrame:
    frame: np.ndarray
    annotated_frame: np.ndarray
    frame_number: int
    results: Any
    detections_count: int
    events: List[Event]


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
        
        self.model_handle: Optional[ModelHandle] = None
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
        
        self._frame_buffer: Optional[AsyncFrameBuffer] = None
        self._result_buffer: Optional[AsyncFrameBuffer] = None
        
        self._producer_task: Optional[asyncio.Task] = None
        self._consumer_task: Optional[asyncio.Task] = None
        self._writer_task: Optional[asyncio.Task] = None
        
        self._total_frames: int = 0
        self._processed_frames: int = 0
        self._detections_count: int = 0
        self._fps: float = 30.0
        self._frame_width: int = 0
        self._frame_height: int = 0
        
        self._frame_count_lock = threading.Lock()
        
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        
        self._buffer_size = getattr(settings, 'FRAME_BUFFER_SIZE', 30)
        self._drop_strategy = getattr(settings, 'FRAME_DROP_STRATEGY', DropStrategy.DROP_OLDEST)
        
        logger.info(f"初始化视频检测器: 任务ID={self.task_id}, 模型={self.model_name}")
    
    def _load_model(self) -> ModelHandle:
        try:
            logger.info(f"加载模型: {self.model_name}")
            handle = device_manager.get_model(self.model_name)
            logger.info(f"模型加载完成: {self.model_name}")
            return handle
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
            self.model_handle = self._load_model()
            
            video_source = self._get_video_source()
            logger.info(f"订阅视频源: {video_source}")
            
            self._frame_buffer = AsyncFrameBuffer(
                max_size=self._buffer_size,
                drop_strategy=self._drop_strategy,
                name=f"frame_buffer_{self.task_id}"
            )
            
            self._result_buffer = AsyncFrameBuffer(
                max_size=self._buffer_size * 2,
                drop_strategy=DropStrategy.DROP_OLDEST,
                name=f"result_buffer_{self.task_id}"
            )
            
            reader = video_source_manager.subscribe(
                task_id=self.task_id,
                source_path=video_source,
                callback=self._on_frame_received
            )
            
            if reader.source_info:
                self._fps = reader.source_info.fps
                self._frame_width = reader.source_info.width
                self._frame_height = reader.source_info.height
                self._total_frames = reader.source_info.total_frames
                
                self._init_video_writer(self._frame_width, self._frame_height, self._fps)
            
            logger.info(f"启动处理任务: task={self.task_id}")
            
            self._consumer_task = asyncio.create_task(self._consumer_loop())
            self._writer_task = asyncio.create_task(self._writer_loop())
            
            await self._wait_for_completion()
            
            logger.info(f"视频处理完成: 总帧数={self._total_frames}, 处理帧数={self._processed_frames}, 检测数量={self._detections_count}, 截图数量={self.screenshot_count}")
            
            EventManager.clear_task_cache(self.task_id)
            
            return True
            
        except asyncio.CancelledError:
            logger.info(f"任务 {self.task_id} 被取消")
            await self._cleanup()
            return False
        except Exception as e:
            import traceback
            error_msg = f"视频处理错误: {str(e)}"
            logger.error(error_msg)
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            await self._cleanup()
            if self.on_error:
                # 界面显示简略错误信息
                simple_error_msg = "视频处理过程中出现错误，请查看日志了解详情"
                self.on_error(self.task_id, simple_error_msg)
            return False
    
    def _on_frame_received(self, video_frame: VideoFrame):
        if not self._is_running or self._should_stop:
            return
        
        if self._should_skip_frame(video_frame.frame_number):
            return
        
        asyncio.create_task(self._frame_buffer.put(video_frame))
    
    async def _consumer_loop(self):
        logger.info(f"消费者循环启动: task={self.task_id}")
        
        while self._is_running and not self._should_stop:
            try:
                video_frame = await self._frame_buffer.get(timeout=1.0)
                
                if video_frame is None:
                    continue
                
                results = await self._process_frame(video_frame.frame, video_frame.frame_number)
                
                annotated_frame = self._draw_detections(video_frame.frame, results, video_frame.frame_number)
                
                detections_count = 0
                events: List[Event] = []
                
                if results and len(results) > 0:
                    detections_count = self._count_people_detections(results)
                    
                    with self._frame_count_lock:
                        self._detections_count += detections_count
                    
                    if self.on_detection:
                        self.on_detection(self.task_id, detections_count)
                    
                    if self.enable_event_generation:
                        events = await self._generate_events(results, video_frame.frame_number, annotated_frame)
                
                processed_frame = ProcessedFrame(
                    frame=video_frame.frame,
                    annotated_frame=annotated_frame,
                    frame_number=video_frame.frame_number,
                    results=results,
                    detections_count=detections_count,
                    events=events
                )
                
                await self._result_buffer.put(processed_frame)
                
                with self._frame_count_lock:
                    self._processed_frames += 1
                
                self._update_progress(video_frame.frame_number)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                import traceback
                logger.error(f"消费者循环错误: {str(e)}")
                logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        await self._result_buffer.close()
        logger.info(f"消费者循环结束: task={self.task_id}")
    
    async def _writer_loop(self):
        logger.info(f"写入器循环启动: task={self.task_id}")
        
        while self._is_running and not self._should_stop:
            try:
                processed_frame = await self._result_buffer.get(timeout=1.0)
                
                if processed_frame is None:
                    continue
                
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame.annotated_frame)
                
            except asyncio.TimeoutError:
                if self._result_buffer.is_closed():
                    break
                continue
            except Exception as e:
                import traceback
                logger.error(f"写入器循环错误: {str(e)}")
                logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        self._close_video_writer()
        logger.info(f"写入器循环结束: task={self.task_id}")
    
    async def _process_frame(self, frame: np.ndarray, frame_count: int) -> list:
        try:
            if self.model_handle is None:
                return []
            
            model = self.model_handle.model
            
            # 处理task_type可能是字符串的情况
            task_type = self.task.task_type
            if hasattr(task_type, 'value'):
                task_type = task_type.value
            
            if task_type == TaskType.PEDESTRIAN_TRACKING.value:
                results = model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=settings.MODEL_IOU_THRESHOLD,
                    verbose=False,
                    persist=True
                )
            else:
                results = model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=settings.MODEL_IOU_THRESHOLD,
                    verbose=False
                )
            
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"处理帧 {frame_count} 时出错: {str(e)}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
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
    
    async def _generate_events(self, results, frame_number: int, annotated_frame: Optional[np.ndarray] = None) -> List[Event]:
        events: List[Event] = []
        
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
                    
                    # 处理task_type可能是字符串的情况
                    task_type_value = self.task.task_type
                    if hasattr(task_type_value, 'value'):
                        task_type_value = task_type_value.value
                    
                    event = self.event_manager.create_event(
                        event_type=event_type,
                        frame_number=frame_number,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        track_id=track_id,
                        metadata={
                            "task_type": task_type_value,
                            "model_name": self.model_name
                        }
                    )
                    
                    if event:
                        events.append(event)
                        
                        if self.on_event:
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
            import traceback
            logger.error(f"生成事件时出错: {str(e)}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
        
        return events
    
    def _get_event_type(self) -> EventType:
        # 处理task_type可能是字符串的情况
        task_type = self.task.task_type
        if hasattr(task_type, 'value'):
            task_type = task_type.value
        
        if task_type == TaskType.PEDESTRIAN_TRACKING.value:
            return EventType.PEDESTRIAN_TRACKED
        return EventType.PEDESTRIAN_DETECTED
    
    def _update_progress(self, current_frame: int):
        if self._total_frames > 0:
            progress = (current_frame / self._total_frames) * 100
        else:
            progress = (self._processed_frames % 100)
        
        if self.on_progress:
            self.on_progress(
                self.task_id,
                min(progress, 100.0),
                current_frame,
                self._total_frames if self._total_frames > 0 else None,
                self._fps if self._fps > 0 else None
            )
    
    async def _wait_for_completion(self):
        while self._is_running and not self._should_stop:
            reader_info = video_source_manager.get_reader_info(self._get_video_source())
            
            if reader_info and not reader_info.is_live:
                with self._frame_count_lock:
                    if self._processed_frames >= self._total_frames and self._total_frames > 0:
                        break
            
            buffer_empty = await self._frame_buffer.is_empty()
            result_empty = await self._result_buffer.is_empty()
            
            if buffer_empty and result_empty and self._processed_frames > 0:
                await asyncio.sleep(0.5)
                
                buffer_empty = await self._frame_buffer.is_empty()
                result_empty = await self._result_buffer.is_empty()
                
                if buffer_empty and result_empty:
                    if reader_info and not reader_info.is_live:
                        break
            
            await asyncio.sleep(0.1)
        
        self._should_stop = True
        
        if self._consumer_task and not self._consumer_task.done():
            try:
                await asyncio.wait_for(self._consumer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"消费者任务超时: task={self.task_id}")
        
        if self._writer_task and not self._writer_task.done():
            try:
                await asyncio.wait_for(self._writer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"写入器任务超时: task={self.task_id}")
    
    async def _cleanup(self):
        self._should_stop = True
        self._is_running = False
        
        video_source_manager.unsubscribe(self.task_id)
        
        if self._frame_buffer:
            await self._frame_buffer.close()
        
        if self._result_buffer:
            await self._result_buffer.close()
        
        self._close_video_writer()
        
        if self.model_handle:
            device_manager.release_model(self.model_handle)
            self.model_handle = None
        
        logger.info(f"任务清理完成: task={self.task_id}")
    
    def stop(self):
        self._should_stop = True
        self._is_running = False
        logger.info(f"停止视频检测器: 任务ID={self.task_id}")
