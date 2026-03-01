"""Central error handling and logging configuration."""
from typing import Dict, Any, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger

# Configure logging
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName

def setup_logging(log_dir: Optional[str] = None) -> None:
    """Set up application-wide logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatters
    json_formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(module)s %(function)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir specified)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_path / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

class AppError(Exception):
    """Base application error class."""
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class ValidationError(AppError):
    """Validation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )

class NotFoundError(AppError):
    """Resource not found error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details
        )

class DatabaseError(AppError):
    """Database operation error."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for all exceptions."""
    logger = logging.getLogger(__name__)
    
    if isinstance(exc, AppError):
        logger.error(f"Application error: {exc.message}", exc_info=True, extra=exc.details)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "status_code": exc.status_code
            }
        )
    
    if isinstance(exc, RequestValidationError):
        logger.error("Validation error", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY
            }
        )
    
    # Log unexpected errors with full traceback
    logger.critical(
        "Unexpected error",
        exc_info=True,
        extra={
            "traceback": traceback.format_exc(),
            "request_path": str(request.url),
            "request_method": request.method
        }
    )
    
    # Return sanitized error response in production
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An unexpected error occurred",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR
        }
    )

def setup_error_handlers(app):
    """Configure FastAPI error handlers."""
    app.add_exception_handler(Exception, error_handler)
    app.add_exception_handler(RequestValidationError, error_handler)
    app.add_exception_handler(AppError, error_handler)