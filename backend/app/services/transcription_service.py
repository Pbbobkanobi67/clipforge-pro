"""Transcription service using faster-whisper."""

import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TranscriptionService:
    """Service for audio transcription using faster-whisper."""

    def __init__(self):
        self._model = None
        self._model_name = None

    def _get_model(self, model_name: str = None):
        """Lazy load the Whisper model."""
        model_name = model_name or settings.whisper_model

        if self._model is None or self._model_name != model_name:
            from faster_whisper import WhisperModel

            logger.info(f"Loading Whisper model: {model_name}")
            self._model = WhisperModel(
                model_name,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            self._model_name = model_name
            logger.info("Whisper model loaded")

        return self._model

    async def transcribe(
        self,
        audio_path: str,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
        word_timestamps: bool = True,
    ) -> dict:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file (WAV format, 16kHz recommended)
            model_name: Whisper model to use (default from settings)
            language: Language code (auto-detect if None)
            word_timestamps: Include word-level timestamps

        Returns:
            Dictionary with transcription results
        """
        import asyncio

        model = self._get_model(model_name)

        # Run transcription in thread pool (CPU-bound)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._transcribe_sync(
                model, audio_path, language, word_timestamps
            ),
        )

        return result

    def _transcribe_sync(
        self,
        model,
        audio_path: str,
        language: Optional[str],
        word_timestamps: bool,
    ) -> dict:
        """Synchronous transcription (for thread pool)."""
        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=True,  # Filter out non-speech
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        # Convert generator to list and process
        processed_segments = []
        full_text_parts = []
        word_count = 0

        for segment in segments:
            words = None
            if word_timestamps and segment.words:
                words = [
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in segment.words
                ]
                word_count += len(words)
            else:
                word_count += len(segment.text.split())

            processed_segments.append({
                "start_time": segment.start,
                "end_time": segment.end,
                "text": segment.text.strip(),
                "words": words,
                "avg_log_prob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            })
            full_text_parts.append(segment.text.strip())

        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "full_text": " ".join(full_text_parts),
            "word_count": word_count,
            "segments": processed_segments,
        }

    def transcribe_with_progress(
        self,
        audio_path: str,
        progress_callback=None,
        model_name: Optional[str] = None,
        language: Optional[str] = None,
    ):
        """
        Transcribe with progress updates.

        Args:
            audio_path: Path to audio file
            progress_callback: Callable(progress: float, message: str)
            model_name: Whisper model to use
            language: Language code

        Yields:
            Segments as they are processed
        """
        model = self._get_model(model_name)

        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )

        duration = info.duration
        processed_segments = []
        full_text_parts = []

        for segment in segments:
            # Calculate progress
            progress = (segment.end / duration) * 100 if duration > 0 else 0

            if progress_callback:
                progress_callback(progress, f"Transcribing: {segment.text[:50]}...")

            words = None
            if segment.words:
                words = [
                    {
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                    for w in segment.words
                ]

            seg_data = {
                "start_time": segment.start,
                "end_time": segment.end,
                "text": segment.text.strip(),
                "words": words,
                "avg_log_prob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
            }

            processed_segments.append(seg_data)
            full_text_parts.append(segment.text.strip())

            yield seg_data

        # Return final result
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "full_text": " ".join(full_text_parts),
            "word_count": sum(
                len(s.get("words", []) or s["text"].split())
                for s in processed_segments
            ),
            "segments": processed_segments,
        }


# Singleton instance for model reuse
_transcription_service = None


def get_transcription_service() -> TranscriptionService:
    """Get singleton transcription service."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service
