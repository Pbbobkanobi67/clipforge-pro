"""Object detection model loader and utilities."""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ObjectDetectionModelLoader:
    """
    Singleton loader for YOLO object detection model.

    Handles model loading and GPU management.
    """

    _instance: Optional["ObjectDetectionModelLoader"] = None
    _model = None
    _model_name: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_name: Optional[str] = None, force_reload: bool = False):
        """
        Load the YOLO model.

        Args:
            model_name: Model name (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            force_reload: Force model reload

        Returns:
            Loaded YOLO model instance
        """
        model_name = model_name or settings.yolo_model

        if self._model is not None and self._model_name == model_name and not force_reload:
            return self._model

        if self._model is not None:
            self.unload()

        from ultralytics import YOLO

        logger.info(f"Loading YOLO model: {model_name}")

        self._model = YOLO(model_name)
        self._model_name = model_name

        logger.info(f"YOLO model {model_name} loaded successfully")
        return self._model

    def unload(self):
        """Unload model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("YOLO model unloaded")

    def get_model(self):
        """Get loaded model or load if not already loaded."""
        if self._model is None:
            return self.load()
        return self._model

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    @property
    def current_model_name(self) -> Optional[str]:
        """Get name of currently loaded model."""
        return self._model_name

    @staticmethod
    def get_available_models() -> list[dict]:
        """Get list of available YOLO models."""
        return [
            {
                "name": "yolov8n.pt",
                "description": "Nano - Fastest, lowest accuracy",
                "params": "3.2M",
                "map": "37.3",
            },
            {
                "name": "yolov8s.pt",
                "description": "Small - Good balance",
                "params": "11.2M",
                "map": "44.9",
            },
            {
                "name": "yolov8m.pt",
                "description": "Medium - Higher accuracy",
                "params": "25.9M",
                "map": "50.2",
            },
            {
                "name": "yolov8l.pt",
                "description": "Large - Very accurate",
                "params": "43.7M",
                "map": "52.9",
            },
            {
                "name": "yolov8x.pt",
                "description": "Extra Large - Best accuracy",
                "params": "68.2M",
                "map": "53.9",
            },
        ]

    @staticmethod
    def estimate_memory_usage(model_name: str) -> dict:
        """Estimate GPU memory usage for a model."""
        memory_map = {
            "yolov8n.pt": 0.5,
            "yolov8s.pt": 0.8,
            "yolov8m.pt": 1.5,
            "yolov8l.pt": 2.5,
            "yolov8x.pt": 4.0,
        }
        return {
            "model": model_name,
            "estimated_vram_gb": memory_map.get(model_name, 2.0),
        }

    def get_class_names(self) -> list[str]:
        """Get list of class names the model can detect."""
        model = self.get_model()
        return list(model.names.values())


# Global instance
yolo_loader = ObjectDetectionModelLoader()


def get_yolo_model(model_name: Optional[str] = None):
    """Convenience function to get YOLO model."""
    return yolo_loader.load(model_name)
