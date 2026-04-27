import logging
import os
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from typing import Optional

from app.config import settings, ensure_directories


class LogCleanupHandler(TimedRotatingFileHandler):
    def __init__(
        self,
        filename: str,
        when: str = 'D',
        interval: int = 1,
        backupCount: int = 7,
        encoding: Optional[str] = 'utf-8',
        delay: bool = False,
        utc: bool = False,
        atTime: Optional[datetime] = None
    ):
        self.retention_days = settings.LOG_RETENTION_DAYS
        super().__init__(
            filename=filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            utc=utc,
            atTime=atTime
        )
    
    def getFilesToDelete(self) -> list:
        dirName, baseName = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        
        prefix = baseName + "."
        plen = len(prefix)
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                suffix = fileName[plen:]
                try:
                    file_date = datetime.strptime(suffix, self.suffix)
                    if file_date < cutoff_date:
                        result.append(os.path.join(dirName, fileName))
                except ValueError:
                    continue
        
        if len(result) < self.backupCount:
            return []
        else:
            result.sort()
            return result[:len(result) - self.backupCount]


def setup_logging():
    ensure_directories()
    
    log_format = (
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
    )
    date_format = '%Y-%m-%d %H:%M:%S'
    
    formatter = logging.Formatter(log_format, date_format)
    
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    log_file = os.path.join(settings.LOG_DIR, 'app.log')
    file_handler = LogCleanupHandler(
        filename=log_file,
        when='D',
        interval=1,
        backupCount=settings.LOG_RETENTION_DAYS,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('cv2').setLevel(logging.WARNING)
    logging.getLogger('fastapi').setLevel(logging.INFO)
    logging.getLogger('uvicorn').setLevel(logging.INFO)
    
    root_logger.info(f"日志系统初始化完成，日志保留天数: {settings.LOG_RETENTION_DAYS}")
    root_logger.info(f"日志文件位置: {log_file}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
