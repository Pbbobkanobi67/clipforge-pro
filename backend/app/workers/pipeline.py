"""Analysis pipeline utilities and helpers."""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineStage:
    """Represents a stage in the analysis pipeline."""

    name: str
    handler: Callable
    progress_start: float
    progress_end: float
    required: bool = True
    depends_on: list[str] = field(default_factory=list)


class AnalysisPipeline:
    """
    Configurable analysis pipeline.

    Allows dynamic stage configuration and execution order management.
    """

    def __init__(self):
        self.stages: dict[str, PipelineStage] = {}
        self.results: dict[str, Any] = {}
        self.progress_callback: Optional[Callable[[float, str], None]] = None

    def add_stage(
        self,
        name: str,
        handler: Callable,
        progress_start: float,
        progress_end: float,
        required: bool = True,
        depends_on: Optional[list[str]] = None,
    ):
        """Add a stage to the pipeline."""
        self.stages[name] = PipelineStage(
            name=name,
            handler=handler,
            progress_start=progress_start,
            progress_end=progress_end,
            required=required,
            depends_on=depends_on or [],
        )

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def _update_progress(self, progress: float, message: str):
        """Update progress via callback if set."""
        if self.progress_callback:
            self.progress_callback(progress, message)

    def _get_execution_order(self, enabled_stages: set[str]) -> list[str]:
        """Determine execution order based on dependencies."""
        order = []
        visited = set()

        def visit(stage_name: str):
            if stage_name in visited:
                return
            if stage_name not in enabled_stages:
                return

            visited.add(stage_name)
            stage = self.stages.get(stage_name)
            if stage:
                for dep in stage.depends_on:
                    visit(dep)
                order.append(stage_name)

        for stage_name in enabled_stages:
            visit(stage_name)

        return order

    async def execute(
        self,
        config: dict,
        initial_data: dict,
    ) -> dict:
        """
        Execute the pipeline.

        Args:
            config: Pipeline configuration (which stages to enable)
            initial_data: Initial data to pass to stages

        Returns:
            Dictionary with all stage results
        """
        # Determine which stages to run
        enabled_stages = set()
        for stage_name, stage in self.stages.items():
            config_key = f"enable_{stage_name}"
            if config.get(config_key, stage.required):
                enabled_stages.add(stage_name)

        # Get execution order
        execution_order = self._get_execution_order(enabled_stages)

        # Execute stages
        self.results = {"_initial": initial_data}

        for stage_name in execution_order:
            stage = self.stages[stage_name]

            self._update_progress(stage.progress_start, f"Starting {stage.name}")

            try:
                # Gather inputs from dependencies
                stage_input = {**initial_data}
                for dep in stage.depends_on:
                    if dep in self.results:
                        stage_input[dep] = self.results[dep]

                # Execute handler
                result = await stage.handler(stage_input, config)
                self.results[stage_name] = result

                self._update_progress(stage.progress_end, f"Completed {stage.name}")

            except Exception as e:
                logger.exception(f"Stage {stage_name} failed: {e}")
                if stage.required:
                    raise
                self.results[stage_name] = {"error": str(e)}

        return self.results


def create_default_pipeline() -> AnalysisPipeline:
    """Create the default video analysis pipeline."""
    pipeline = AnalysisPipeline()

    # Add stages in dependency order
    pipeline.add_stage(
        name="audio_extraction",
        handler=_audio_extraction_handler,
        progress_start=0,
        progress_end=5,
        required=True,
    )

    pipeline.add_stage(
        name="transcription",
        handler=_transcription_handler,
        progress_start=5,
        progress_end=30,
        depends_on=["audio_extraction"],
    )

    pipeline.add_stage(
        name="diarization",
        handler=_diarization_handler,
        progress_start=30,
        progress_end=45,
        required=False,
        depends_on=["audio_extraction"],
    )

    pipeline.add_stage(
        name="scene_detection",
        handler=_scene_detection_handler,
        progress_start=45,
        progress_end=55,
        required=False,
    )

    pipeline.add_stage(
        name="visual_analysis",
        handler=_visual_analysis_handler,
        progress_start=55,
        progress_end=70,
        required=False,
    )

    pipeline.add_stage(
        name="hook_detection",
        handler=_hook_detection_handler,
        progress_start=70,
        progress_end=80,
        required=False,
        depends_on=["transcription"],
    )

    pipeline.add_stage(
        name="clip_suggestion",
        handler=_clip_suggestion_handler,
        progress_start=80,
        progress_end=95,
        depends_on=["transcription", "scene_detection", "hook_detection"],
    )

    return pipeline


async def _audio_extraction_handler(input_data: dict, config: dict) -> dict:
    """Handler for audio extraction stage."""
    from app.services.video_service import VideoService
    from pathlib import Path
    from app.config import get_settings

    settings = get_settings()
    video_service = VideoService()

    video_path = input_data["video_path"]
    job_id = input_data["job_id"]

    audio_path = str(Path(settings.cache_path) / f"{job_id}.wav")
    await video_service.extract_audio(video_path, audio_path)

    return {"audio_path": audio_path}


async def _transcription_handler(input_data: dict, config: dict) -> dict:
    """Handler for transcription stage."""
    from app.services.transcription_service import get_transcription_service

    service = get_transcription_service()
    audio_path = input_data["audio_extraction"]["audio_path"]

    result = await service.transcribe(
        audio_path,
        model_name=config.get("whisper_model"),
    )

    return result


async def _diarization_handler(input_data: dict, config: dict) -> dict:
    """Handler for diarization stage."""
    from app.services.diarization_service import get_diarization_service

    service = get_diarization_service()
    audio_path = input_data["audio_extraction"]["audio_path"]

    result = await service.diarize(audio_path)

    # Merge with transcription if available
    if "transcription" in input_data:
        transcription = input_data["transcription"]
        result["merged_segments"] = service.merge_with_transcription(
            transcription.get("segments", []),
            result.get("segments", []),
        )

    return result


async def _scene_detection_handler(input_data: dict, config: dict) -> dict:
    """Handler for scene detection stage."""
    from app.services.scene_detection_service import get_scene_detection_service
    from pathlib import Path
    from app.config import get_settings

    settings = get_settings()
    service = get_scene_detection_service()

    video_path = input_data["video_path"]
    job_id = input_data["job_id"]

    keyframe_dir = str(Path(settings.cache_path) / job_id)
    Path(keyframe_dir).mkdir(parents=True, exist_ok=True)

    result = await service.detect_scenes(
        video_path,
        extract_keyframes=True,
        keyframe_dir=keyframe_dir,
    )

    return result


async def _visual_analysis_handler(input_data: dict, config: dict) -> dict:
    """Handler for visual analysis stage."""
    from app.services.visual_analysis_service import get_visual_analysis_service

    service = get_visual_analysis_service()
    video_path = input_data["video_path"]

    result = await service.analyze_video(
        video_path,
        sample_interval=config.get("visual_sample_interval", 2.0),
    )

    return result


async def _hook_detection_handler(input_data: dict, config: dict) -> dict:
    """Handler for hook detection stage."""
    from app.services.hook_detection_service import get_hook_detection_service

    service = get_hook_detection_service()
    transcription = input_data.get("transcription", {})
    visual_data = input_data.get("visual_analysis")

    hooks = await service.detect_hooks(
        transcription.get("segments", []),
        visual_analysis=visual_data,
    )

    return {"hooks": hooks}


async def _clip_suggestion_handler(input_data: dict, config: dict) -> dict:
    """Handler for clip suggestion stage."""
    from app.services.clip_suggestion_service import get_clip_suggestion_service
    from app.services.llm_service import get_llm_service
    from app.services.video_service import VideoService

    service = get_clip_suggestion_service()
    video_service = VideoService()

    video_path = input_data["video_path"]
    video_metadata = await video_service.get_video_metadata(video_path)

    llm_service = None
    if config.get("llm_analysis", True):
        llm_service = get_llm_service()

    clips = await service.suggest_clips(
        transcription_data=input_data.get("transcription", {}),
        scene_data=input_data.get("scene_detection"),
        hook_data=input_data.get("hook_detection", {}).get("hooks"),
        visual_data=input_data.get("visual_analysis"),
        diarization_data=input_data.get("diarization"),
        video_metadata=video_metadata,
        min_duration=config.get("min_clip_duration", 15.0),
        max_duration=config.get("max_clip_duration", 90.0),
        target_count=config.get("target_clips", 5),
        llm_service=llm_service,
    )

    return {"clips": clips}
