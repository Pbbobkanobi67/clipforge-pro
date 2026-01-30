"""Whisper model loader and utilities."""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class WhisperModelLoader:
    """
    Singleton loader for faster-whisper model.

    Handles model loading, caching, and GPU memory management.
    """

    _instance: Optional["WhisperModelLoader"] = None
    _model = None
    _model_name: Optional[str] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, model_name: Optional[str] = None, force_reload: bool = False):
        """
        Load the Whisper model.

        Args:
            model_name: Model name (tiny, base, small, medium, large-v2, large-v3)
            force_reload: Force model reload even if already loaded

        Returns:
            Loaded WhisperModel instance
        """
        model_name = model_name or settings.whisper_model

        if self._model is not None and self._model_name == model_name and not force_reload:
            return self._model

        # Unload existing model to free GPU memory
        if self._model is not None:
            self.unload()

        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper model: {model_name}")
        logger.info(f"Device: {settings.whisper_device}, Compute: {settings.whisper_compute_type}")

        self._model = WhisperModel(
            model_name,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            download_root=str(settings.cache_path / "whisper_models"),
        )
        self._model_name = model_name

        logger.info(f"Whisper model {model_name} loaded successfully")
        return self._model

    def unload(self):
        """Unload model and free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_name = None

            # Force GPU memory cleanup
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Whisper model unloaded")

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
    def get_available_models() -> list[str]:
        """Get list of available Whisper models."""
        return [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large-v1",
            "large-v2",
            "large-v3",
        ]

    @staticmethod
    def estimate_memory_usage(model_name: str) -> dict:
        """Estimate GPU memory usage for a model."""
        # Approximate VRAM usage in GB (FP16)
        memory_map = {
            "tiny": 1.0,
            "tiny.en": 1.0,
            "base": 1.0,
            "base.en": 1.0,
            "small": 2.0,
            "small.en": 2.0,
            "medium": 5.0,
            "medium.en": 5.0,
            "large-v1": 10.0,
            "large-v2": 10.0,
            "large-v3": 10.0,
        }
        return {
            "model": model_name,
            "estimated_vram_gb": memory_map.get(model_name, 10.0),
            "compute_type": settings.whisper_compute_type,
        }


# Global instance
whisper_loader = WhisperModelLoader()


def get_whisper_model(model_name: Optional[str] = None):
    """Convenience function to get Whisper model."""
    return whisper_loader.load(model_name)
