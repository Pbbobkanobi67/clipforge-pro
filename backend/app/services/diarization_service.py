"""Speaker diarization service using pyannote.audio."""

import logging
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DiarizationService:
    """Service for speaker diarization using pyannote.audio."""

    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        """Lazy load the diarization pipeline."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline
            import torch

            logger.info("Loading pyannote diarization pipeline...")

            # Load pretrained pipeline
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.hf_token,
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))

            logger.info("Diarization pipeline loaded")

        return self._pipeline

    async def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> dict:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers

        Returns:
            Dictionary with diarization results
        """
        import asyncio

        pipeline = self._get_pipeline()

        # Run diarization in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._diarize_sync(
                pipeline, audio_path, num_speakers, min_speakers, max_speakers
            ),
        )

        return result

    def _diarize_sync(
        self,
        pipeline,
        audio_path: str,
        num_speakers: Optional[int],
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict:
        """Synchronous diarization (for thread pool)."""
        # Build parameters
        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        # Run diarization
        diarization = pipeline(audio_path, **params)

        # Process results
        segments = []
        speaker_labels = set()

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_time": turn.start,
                "end_time": turn.end,
                "speaker_id": speaker,
            })
            speaker_labels.add(speaker)

        return {
            "speaker_count": len(speaker_labels),
            "speaker_labels": sorted(list(speaker_labels)),
            "segments": segments,
        }

    def merge_with_transcription(
        self,
        transcription_segments: list[dict],
        diarization_segments: list[dict],
    ) -> list[dict]:
        """
        Merge diarization results with transcription segments.

        Assigns speaker IDs to transcription segments based on overlap.

        Args:
            transcription_segments: List of transcription segments
            diarization_segments: List of diarization segments

        Returns:
            Transcription segments with speaker_id assigned
        """
        merged = []

        for trans_seg in transcription_segments:
            trans_start = trans_seg["start_time"]
            trans_end = trans_seg["end_time"]

            # Find overlapping diarization segment
            best_speaker = None
            best_overlap = 0

            for diar_seg in diarization_segments:
                diar_start = diar_seg["start_time"]
                diar_end = diar_seg["end_time"]

                # Calculate overlap
                overlap_start = max(trans_start, diar_start)
                overlap_end = min(trans_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = diar_seg["speaker_id"]

            merged_seg = {**trans_seg, "speaker_id": best_speaker}
            merged.append(merged_seg)

        return merged


# Singleton instance
_diarization_service = None


def get_diarization_service() -> DiarizationService:
    """Get singleton diarization service."""
    global _diarization_service
    if _diarization_service is None:
        _diarization_service = DiarizationService()
    return _diarization_service
