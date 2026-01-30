"""Speaker diarization model loader and utilities."""

import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DiarizationModelLoader:
    """
    Singleton loader for pyannote diarization pipeline.

    Handles model loading, authentication, and GPU management.
    """

    _instance: Optional["DiarizationModelLoader"] = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, force_reload: bool = False):
        """
        Load the diarization pipeline.

        Requires HuggingFace token for pyannote models.

        Args:
            force_reload: Force pipeline reload

        Returns:
            Loaded Pipeline instance
        """
        if self._pipeline is not None and not force_reload:
            return self._pipeline

        if self._pipeline is not None:
            self.unload()

        if not settings.hf_token:
            raise ValueError(
                "HuggingFace token required for pyannote models. "
                "Set HF_TOKEN environment variable."
            )

        from pyannote.audio import Pipeline
        import torch

        logger.info("Loading pyannote speaker diarization pipeline...")

        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            self._pipeline.to(torch.device("cuda"))
            logger.info("Diarization pipeline moved to GPU")

        logger.info("Diarization pipeline loaded successfully")
        return self._pipeline

    def unload(self):
        """Unload pipeline and free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Diarization pipeline unloaded")

    def get_pipeline(self):
        """Get loaded pipeline or load if not already loaded."""
        if self._pipeline is None:
            return self.load()
        return self._pipeline

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is currently loaded."""
        return self._pipeline is not None

    @staticmethod
    def get_available_models() -> list[dict]:
        """Get list of available diarization models."""
        return [
            {
                "name": "pyannote/speaker-diarization-3.1",
                "description": "Latest speaker diarization pipeline",
                "requires_auth": True,
            },
            {
                "name": "pyannote/speaker-diarization-2.1",
                "description": "Previous version, faster but less accurate",
                "requires_auth": True,
            },
        ]

    @staticmethod
    def estimate_memory_usage() -> dict:
        """Estimate GPU memory usage for diarization."""
        return {
            "model": "pyannote/speaker-diarization-3.1",
            "estimated_vram_gb": 2.0,
            "notes": "Memory usage scales with audio duration",
        }


# Global instance
diarization_loader = DiarizationModelLoader()


def get_diarization_pipeline():
    """Convenience function to get diarization pipeline."""
    return diarization_loader.load()
