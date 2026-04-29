import os
from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "YOLO视频检测系统"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # 管理员配置
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "123456"
    
    # YOLO模型配置
    DEFAULT_MODEL: str = "yolo11n.pt"
    AVAILABLE_MODELS: List[str] = [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
    ]
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5
    MODEL_IOU_THRESHOLD: float = 0.5
    
    # 视频处理配置
    DEFAULT_FRAME_SKIP: int = 5
    MIN_FRAME_SKIP: int = 1
    MAX_FRAME_SKIP: int = 30
    
    # 事件配置
    EVENT_COOLDOWN_SECONDS: int = 10
    EVENT_MIN_DURATION_SECONDS: float = 1.0
    EVENT_MAX_DURATION_SECONDS: float = 60.0
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    LOG_RETENTION_DAYS: int = 7
    LOG_MAX_SIZE_MB: int = 10
    LOG_BACKUP_COUNT: int = 5
    
    # 任务配置
    MAX_CONCURRENT_TASKS: int = 5
    TASK_TIMEOUT_SECONDS: int = 3600
    
    # 帧缓冲配置
    FRAME_BUFFER_SIZE: int = 30
    FRAME_DROP_STRATEGY: str = "drop_oldest"
    
    # 推理引擎配置
    INFERENCE_BATCH_SIZE: int = 4
    INFERENCE_MAX_QUEUE: int = 100
    INFERENCE_BATCH_TIMEOUT: int = 100
    
    # 设备配置
    FORCE_CPU: bool = False
    
    # 数据存储路径
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    EVENTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events")
    TASKS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tasks")
    VIDEO_OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "videos")
    SCREENSHOTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "screenshots")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


# 确保必要目录存在
def ensure_directories():
    directories = [
        settings.LOG_DIR,
        settings.DATA_DIR,
        settings.EVENTS_DIR,
        settings.TASKS_DIR,
        settings.VIDEO_OUTPUT_DIR,
        settings.SCREENSHOTS_DIR
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
