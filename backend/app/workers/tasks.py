"""Celery tasks for video analysis."""

import logging
import uuid
from datetime import datetime
from pathlib import Path

from celery import shared_task

from app.config import get_settings
from app.models.database import (
    AnalysisJob,
    AnalysisStatus,
    Video,
    Transcription,
    TranscriptionSegment,
    Diarization,
    DiarizationSegment,
    Scene,
    ClipSuggestion,
    Hook,
    SyncSessionLocal,
)

logger = logging.getLogger(__name__)
settings = get_settings()


def update_job_status(
    session,
    job_id: str,
    status: AnalysisStatus,
    progress: float,
    stage: str = None,
    error: str = None,
):
    """Update job status in database."""
    job = session.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
    if job:
        job.status = status
        job.progress = progress
        if stage:
            job.current_stage = stage
        if error:
            job.error_message = error
        if status == AnalysisStatus.COMPLETED:
            job.completed_at = datetime.utcnow()
        session.commit()


@shared_task(bind=True, name="app.workers.tasks.run_analysis_pipeline")
def run_analysis_pipeline(self, job_id: str):
    """
    Main analysis pipeline task.

    Orchestrates all analysis stages:
    1. Audio extraction
    2. Transcription
    3. Diarization
    4. Scene detection
    5. Visual analysis
    6. Hook detection
    7. Clip suggestion
    8. Virality scoring
    """
    import asyncio

    session = SyncSessionLocal()

    try:
        # Get job and video
        job = session.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
        if not job:
            logger.error(f"Job not found: {job_id}")
            return

        video = session.query(Video).filter(Video.id == job.video_id).first()
        if not video:
            logger.error(f"Video not found for job: {job_id}")
            update_job_status(session, job_id, AnalysisStatus.FAILED, 0, error="Video not found")
            return

        config = job.config or {}
        video_path = video.file_path

        logger.info(f"Starting analysis pipeline for job {job_id}, video: {video_path}")

        # Run the async pipeline
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _run_pipeline_async(session, job_id, video_path, config)
            )
        finally:
            loop.close()

        if result.get("success"):
            update_job_status(session, job_id, AnalysisStatus.COMPLETED, 100, "Completed")
            logger.info(f"Analysis pipeline completed for job {job_id}")
        else:
            update_job_status(
                session, job_id, AnalysisStatus.FAILED, result.get("progress", 0),
                error=result.get("error", "Unknown error")
            )

    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}: {e}")
        update_job_status(session, job_id, AnalysisStatus.FAILED, 0, error=str(e))
    finally:
        session.close()


async def _run_pipeline_async(
    session,
    job_id: str,
    video_path: str,
    config: dict,
) -> dict:
    """Async implementation of the analysis pipeline."""
    from app.services.video_service import VideoService
    from app.services.transcription_service import get_transcription_service
    from app.services.diarization_service import get_diarization_service
    from app.services.scene_detection_service import get_scene_detection_service
    from app.services.visual_analysis_service import get_visual_analysis_service
    from app.services.hook_detection_service import get_hook_detection_service
    from app.services.clip_suggestion_service import get_clip_suggestion_service
    from app.services.virality_service import get_virality_service
    from app.services.llm_service import get_llm_service

    video_service = VideoService()
    transcription_service = get_transcription_service()
    diarization_service = get_diarization_service()
    scene_service = get_scene_detection_service()
    visual_service = get_visual_analysis_service()
    hook_service = get_hook_detection_service()
    clip_service = get_clip_suggestion_service()
    llm_service = get_llm_service() if config.get("llm_analysis", True) else None

    results = {
        "transcription": None,
        "diarization": None,
        "scenes": None,
        "visual": None,
        "hooks": None,
        "clips": None,
    }

    try:
        # Stage 1: Extract audio (5%)
        update_job_status(session, job_id, AnalysisStatus.PROCESSING, 5, "Extracting audio")
        audio_path = str(Path(settings.cache_path) / f"{job_id}.wav")
        await video_service.extract_audio(video_path, audio_path)

        # Stage 2: Transcription (5-30%)
        if config.get("transcribe", True):
            update_job_status(session, job_id, AnalysisStatus.TRANSCRIBING, 10, "Transcribing")
            transcription_result = await transcription_service.transcribe(
                audio_path,
                model_name=config.get("whisper_model", settings.whisper_model),
            )
            results["transcription"] = transcription_result

            # Save transcription to database
            _save_transcription(session, job_id, transcription_result)
            update_job_status(session, job_id, AnalysisStatus.TRANSCRIBING, 30, "Transcription complete")

        # Stage 3: Diarization (30-45%)
        if config.get("diarize", True):
            update_job_status(session, job_id, AnalysisStatus.DIARIZING, 35, "Identifying speakers")
            try:
                diarization_result = await diarization_service.diarize(audio_path)
                results["diarization"] = diarization_result

                # Merge with transcription
                if results["transcription"]:
                    results["transcription"]["segments"] = diarization_service.merge_with_transcription(
                        results["transcription"]["segments"],
                        diarization_result["segments"],
                    )
                    # Update transcription segments with speaker IDs
                    _update_transcription_speakers(session, job_id, results["transcription"]["segments"])

                _save_diarization(session, job_id, diarization_result)
            except Exception as e:
                logger.warning(f"Diarization failed: {e}")
            update_job_status(session, job_id, AnalysisStatus.DIARIZING, 45, "Diarization complete")

        # Stage 4: Scene Detection (45-55%)
        if config.get("detect_scenes", True):
            update_job_status(session, job_id, AnalysisStatus.DETECTING_SCENES, 50, "Detecting scenes")
            keyframe_dir = str(Path(settings.cache_path) / job_id)
            Path(keyframe_dir).mkdir(parents=True, exist_ok=True)

            scene_result = await scene_service.detect_scenes(
                video_path,
                extract_keyframes=True,
                keyframe_dir=keyframe_dir,
            )
            results["scenes"] = scene_result
            _save_scenes(session, job_id, scene_result)
            update_job_status(session, job_id, AnalysisStatus.DETECTING_SCENES, 55, "Scene detection complete")

        # Stage 5: Visual Analysis (55-70%)
        if config.get("analyze_visuals", True):
            update_job_status(session, job_id, AnalysisStatus.ANALYZING_VISUALS, 60, "Analyzing visuals")
            visual_result = await visual_service.analyze_video(video_path, sample_interval=2.0)
            results["visual"] = visual_result
            update_job_status(session, job_id, AnalysisStatus.ANALYZING_VISUALS, 70, "Visual analysis complete")

        # Stage 6: Hook Detection (70-80%)
        if config.get("detect_hooks", True) and results["transcription"]:
            update_job_status(session, job_id, AnalysisStatus.DETECTING_HOOKS, 75, "Detecting hooks")
            hooks = await hook_service.detect_hooks(
                results["transcription"]["segments"],
                visual_analysis=results["visual"],
            )
            results["hooks"] = hooks
            _save_hooks(session, job_id, hooks)
            update_job_status(session, job_id, AnalysisStatus.DETECTING_HOOKS, 80, "Hook detection complete")

        # Stage 7: Clip Suggestion (80-95%)
        if config.get("suggest_clips", True) and results["transcription"]:
            update_job_status(session, job_id, AnalysisStatus.SUGGESTING_CLIPS, 85, "Suggesting clips")

            # Get video metadata
            video_metadata = await video_service.get_video_metadata(video_path)

            clips = await clip_service.suggest_clips(
                transcription_data=results["transcription"],
                scene_data=results["scenes"],
                hook_data=results["hooks"],
                visual_data=results["visual"],
                diarization_data=results["diarization"],
                video_metadata=video_metadata,
                min_duration=config.get("min_clip_duration", 15.0),
                max_duration=config.get("max_clip_duration", 90.0),
                target_count=config.get("target_clips", 5),
                llm_service=llm_service,
            )
            results["clips"] = clips
            _save_clips(session, job_id, clips)
            update_job_status(session, job_id, AnalysisStatus.SUGGESTING_CLIPS, 95, "Clip suggestion complete")

        # Update job summary
        _update_job_summary(session, job_id, results)

        # Cleanup temp files
        Path(audio_path).unlink(missing_ok=True)

        return {"success": True, "progress": 100}

    except Exception as e:
        logger.exception(f"Pipeline stage failed: {e}")
        return {"success": False, "error": str(e), "progress": 0}


def _save_transcription(session, job_id: str, data: dict):
    """Save transcription results to database."""
    job_uuid = uuid.UUID(job_id)

    transcription = Transcription(
        analysis_job_id=job_uuid,
        full_text=data["full_text"],
        language=data.get("language"),
        language_probability=data.get("language_probability"),
        word_count=data.get("word_count", 0),
        duration_seconds=data.get("duration_seconds"),
    )
    session.add(transcription)
    session.flush()

    for seg in data.get("segments", []):
        segment = TranscriptionSegment(
            transcription_id=transcription.id,
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            text=seg["text"],
            words=seg.get("words"),
            avg_log_prob=seg.get("avg_log_prob"),
            no_speech_prob=seg.get("no_speech_prob"),
        )
        session.add(segment)

    session.commit()


def _update_transcription_speakers(session, job_id: str, segments: list[dict]):
    """Update transcription segments with speaker IDs."""
    job_uuid = uuid.UUID(job_id)

    transcription = session.query(Transcription).filter(
        Transcription.analysis_job_id == job_uuid
    ).first()

    if not transcription:
        return

    db_segments = session.query(TranscriptionSegment).filter(
        TranscriptionSegment.transcription_id == transcription.id
    ).order_by(TranscriptionSegment.start_time).all()

    for db_seg, data_seg in zip(db_segments, segments):
        db_seg.speaker_id = data_seg.get("speaker_id")

    session.commit()


def _save_diarization(session, job_id: str, data: dict):
    """Save diarization results to database."""
    job_uuid = uuid.UUID(job_id)

    diarization = Diarization(
        analysis_job_id=job_uuid,
        speaker_count=data["speaker_count"],
        speaker_labels=data["speaker_labels"],
    )
    session.add(diarization)
    session.flush()

    for seg in data.get("segments", []):
        segment = DiarizationSegment(
            diarization_id=diarization.id,
            start_time=seg["start_time"],
            end_time=seg["end_time"],
            speaker_id=seg["speaker_id"],
        )
        session.add(segment)

    session.commit()


def _save_scenes(session, job_id: str, data: dict):
    """Save scene detection results to database."""
    job_uuid = uuid.UUID(job_id)

    for scene in data.get("scenes", []):
        db_scene = Scene(
            analysis_job_id=job_uuid,
            start_time=scene["start_time"],
            end_time=scene["end_time"],
            duration=scene["duration"],
            scene_index=scene["scene_index"],
            scene_type=scene.get("scene_type"),
            keyframe_path=scene.get("keyframe_path"),
            dominant_colors=scene.get("dominant_colors"),
            motion_intensity=scene.get("motion_intensity"),
        )
        session.add(db_scene)

    session.commit()


def _save_hooks(session, job_id: str, hooks: list[dict]):
    """Save hook detection results to database."""
    job_uuid = uuid.UUID(job_id)

    for hook in hooks:
        db_hook = Hook(
            analysis_job_id=job_uuid,
            start_time=hook["start_time"],
            end_time=hook["end_time"],
            text=hook.get("text"),
            hook_type=hook.get("hook_type"),
            hook_score=hook["hook_score"],
            curiosity_gap=hook.get("curiosity_gap", 0),
            emotional_trigger=hook.get("emotional_trigger", 0),
            clarity=hook.get("clarity", 0),
            visual_interest=hook.get("visual_interest", 0),
            is_video_start=hook.get("is_video_start", False),
            rank=hook.get("rank"),
        )
        session.add(db_hook)

    session.commit()


def _save_clips(session, job_id: str, clips: list[dict]):
    """Save clip suggestions to database."""
    job_uuid = uuid.UUID(job_id)

    for clip in clips:
        db_clip = ClipSuggestion(
            analysis_job_id=job_uuid,
            start_time=clip["start_time"],
            end_time=clip["end_time"],
            duration=clip["duration"],
            title=clip.get("title"),
            description=clip.get("description"),
            transcript_excerpt=clip.get("transcript_excerpt"),
            virality_score=clip["virality_score"],
            emotional_resonance=clip.get("emotional_resonance", 0),
            shareability=clip.get("shareability", 0),
            uniqueness=clip.get("uniqueness", 0),
            hook_strength=clip.get("hook_strength", 0),
            production_quality=clip.get("production_quality", 0),
            rank=clip.get("rank"),
        )
        session.add(db_clip)

    session.commit()


def _update_job_summary(session, job_id: str, results: dict):
    """Update job with summary statistics."""
    job = session.query(AnalysisJob).filter(AnalysisJob.id == uuid.UUID(job_id)).first()
    if not job:
        return

    if results.get("transcription"):
        job.word_count = results["transcription"].get("word_count", 0)
        job.total_duration = results["transcription"].get("duration_seconds")

    if results.get("diarization"):
        job.speaker_count = results["diarization"].get("speaker_count", 0)

    if results.get("scenes"):
        job.scene_count = results["scenes"].get("scene_count", 0)

    if results.get("clips"):
        job.clip_count = len(results["clips"])

    session.commit()
