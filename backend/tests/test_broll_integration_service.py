"""Tests for BRollIntegrationService."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.broll_integration_service import (
    BRollIntegrationService,
    get_broll_integration_service,
)


class TestBRollIntegrationService:
    """Test cases for BRollIntegrationService."""

    @pytest.fixture
    def service(self) -> BRollIntegrationService:
        """Create a BRollIntegrationService instance."""
        return BRollIntegrationService()

    @pytest.fixture
    def sample_insertions(self, tmp_path: Path) -> list[dict]:
        """Sample B-roll insertions for testing."""
        broll1 = tmp_path / "broll1.mp4"
        broll2 = tmp_path / "broll2.mp4"
        broll1.write_bytes(b"fake broll 1")
        broll2.write_bytes(b"fake broll 2")

        return [
            {
                "start_time": 5.0,
                "duration": 3.0,
                "asset_path": str(broll1),
                "mode": "full_replace",
                "transition": "crossfade",
            },
            {
                "start_time": 15.0,
                "duration": 4.0,
                "asset_path": str(broll2),
                "mode": "full_replace",
                "transition": "fade",
            },
        ]

    @pytest.mark.asyncio
    async def test_apply_broll_no_insertions(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test that empty insertions just copies the file."""
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"
        input_path.write_bytes(b"original video content")

        result = await service.apply_broll(
            input_path=str(input_path),
            output_path=str(output_path),
            broll_insertions=[],
        )

        assert Path(result).exists()
        assert Path(result).read_bytes() == b"original video content"

    @pytest.mark.asyncio
    async def test_apply_broll_sorts_insertions(
        self, service: BRollIntegrationService, sample_insertions: list[dict]
    ):
        """Test that insertions are sorted by start time."""
        # Reverse the order
        reversed_insertions = sample_insertions[::-1]

        with patch.object(service, "_apply_broll_full_replace", new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = "/output.mp4"

            await service.apply_broll(
                input_path="/input.mp4",
                output_path="/output.mp4",
                broll_insertions=reversed_insertions,
            )

            # Check that insertions were sorted
            call_args = mock_apply.call_args
            insertions = call_args[0][2]  # Third positional arg
            assert insertions[0]["start_time"] < insertions[1]["start_time"]

    @pytest.mark.asyncio
    async def test_apply_broll_adjusts_overlapping(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test that overlapping insertions are adjusted."""
        broll = tmp_path / "broll.mp4"
        broll.write_bytes(b"fake broll")

        overlapping_insertions = [
            {
                "start_time": 5.0,
                "duration": 10.0,  # Ends at 15.0
                "asset_path": str(broll),
                "mode": "full_replace",
                "transition": "crossfade",
            },
            {
                "start_time": 10.0,  # Overlaps with first
                "duration": 5.0,
                "asset_path": str(broll),
                "mode": "full_replace",
                "transition": "fade",
            },
        ]

        with patch.object(service, "_apply_broll_full_replace", new_callable=AsyncMock) as mock_apply:
            mock_apply.return_value = "/output.mp4"

            await service.apply_broll(
                input_path="/input.mp4",
                output_path="/output.mp4",
                broll_insertions=overlapping_insertions,
            )

            # Check that second insertion was adjusted
            call_args = mock_apply.call_args
            insertions = call_args[0][2]
            assert insertions[1]["start_time"] >= insertions[0]["start_time"] + insertions[0]["duration"]

    @pytest.mark.asyncio
    async def test_apply_broll_detects_pip_mode(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test that PIP mode is detected and handled."""
        broll = tmp_path / "broll.mp4"
        broll.write_bytes(b"fake broll")

        pip_insertions = [
            {
                "start_time": 5.0,
                "duration": 3.0,
                "asset_path": str(broll),
                "mode": "pip_overlay",
                "transition": "fade",
            },
        ]

        with patch.object(service, "_apply_broll_with_pip", new_callable=AsyncMock) as mock_pip:
            mock_pip.return_value = "/output.mp4"

            await service.apply_broll(
                input_path="/input.mp4",
                output_path="/output.mp4",
                broll_insertions=pip_insertions,
            )

            mock_pip.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_video_info(self, service: BRollIntegrationService):
        """Test video info extraction."""
        mock_output = json.dumps({
            "streams": [{"width": 1920, "height": 1080, "duration": "30.5"}],
            "format": {"duration": "30.5"},
        })

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output)

            info = await service._get_video_info("/test/video.mp4")

        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["duration"] == 30.5

    @pytest.mark.asyncio
    async def test_get_video_info_handles_missing_data(self, service: BRollIntegrationService):
        """Test video info with missing data falls back to defaults."""
        mock_output = json.dumps({"streams": [], "format": {}})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=mock_output)

            info = await service._get_video_info("/test/video.mp4")

        assert info["width"] == 1920  # Default
        assert info["height"] == 1080  # Default
        assert info["duration"] == 0  # Default

    @pytest.mark.asyncio
    async def test_run_ffmpeg_success(self, service: BRollIntegrationService):
        """Test successful FFmpeg execution."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")

            # Should not raise
            await service._run_ffmpeg(["ffmpeg", "-i", "input.mp4", "output.mp4"])

            mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ffmpeg_failure(self, service: BRollIntegrationService):
        """Test FFmpeg execution failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error: invalid input")

            with pytest.raises(RuntimeError) as exc_info:
                await service._run_ffmpeg(["ffmpeg", "-i", "input.mp4", "output.mp4"])

            assert "FFmpeg failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_segment(self, service: BRollIntegrationService):
        """Test segment extraction."""
        with patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:
            await service._extract_segment(
                input_path="/input.mp4",
                output_path="/segment.mp4",
                start_time=5.0,
                duration=10.0,
                target_width=1920,
                target_height=1080,
            )

            mock_ffmpeg.assert_called_once()
            cmd = mock_ffmpeg.call_args[0][0]

            assert "ffmpeg" in cmd
            assert "-ss" in cmd
            assert "5.0" in cmd
            assert "-t" in cmd
            assert "10.0" in cmd

    @pytest.mark.asyncio
    async def test_prepare_broll_segment(self, service: BRollIntegrationService):
        """Test B-roll segment preparation."""
        with patch.object(service, "_get_video_info", new_callable=AsyncMock) as mock_info, \
             patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:

            mock_info.return_value = {"duration": 10.0, "width": 1920, "height": 1080}

            await service._prepare_broll_segment(
                input_path="/broll.mp4",
                output_path="/broll_prepared.mp4",
                target_duration=5.0,
                target_width=1920,
                target_height=1080,
            )

            mock_ffmpeg.assert_called_once()

    @pytest.mark.asyncio
    async def test_prepare_broll_segment_loops_short_video(
        self, service: BRollIntegrationService
    ):
        """Test that short B-roll videos are looped."""
        with patch.object(service, "_get_video_info", new_callable=AsyncMock) as mock_info, \
             patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:

            # B-roll is shorter than target
            mock_info.return_value = {"duration": 3.0, "width": 1920, "height": 1080}

            await service._prepare_broll_segment(
                input_path="/broll.mp4",
                output_path="/broll_prepared.mp4",
                target_duration=10.0,  # Longer than source
                target_width=1920,
                target_height=1080,
            )

            # Should include loop flag
            cmd = mock_ffmpeg.call_args[0][0]
            assert "-stream_loop" in cmd

    @pytest.mark.asyncio
    async def test_concatenate_with_transitions_single_segment(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test concatenation with single segment."""
        segment_file = tmp_path / "segment.mp4"
        segment_file.write_bytes(b"segment content")
        output_path = tmp_path / "output.mp4"

        segments = [{"file": str(segment_file), "type": "main", "duration": 10.0}]

        await service._concatenate_with_transitions(segments, str(output_path), 0.5)

        # Single segment should just be copied
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_concatenate_with_transitions_multiple_segments(
        self, service: BRollIntegrationService
    ):
        """Test concatenation with multiple segments and transitions."""
        segments = [
            {"file": "/seg1.mp4", "type": "main", "duration": 10.0},
            {"file": "/seg2.mp4", "type": "broll", "duration": 5.0, "transition": "crossfade"},
            {"file": "/seg3.mp4", "type": "main", "duration": 15.0},
        ]

        with patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:
            await service._concatenate_with_transitions(segments, "/output.mp4", 0.5)

            cmd = mock_ffmpeg.call_args[0][0]

            # Should include xfade filter
            assert "-filter_complex" in cmd
            filter_idx = cmd.index("-filter_complex")
            filter_str = cmd[filter_idx + 1]
            assert "xfade" in filter_str

    @pytest.mark.asyncio
    async def test_mix_audio(self, service: BRollIntegrationService):
        """Test audio mixing."""
        with patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg:
            await service._mix_audio(
                original_video="/original.mp4",
                video_without_audio="/video_only.mp4",
                output_path="/output.mp4",
            )

            cmd = mock_ffmpeg.call_args[0][0]

            # Should map video from first input and audio from second
            assert "-map" in cmd
            assert "0:v" in cmd
            assert "1:a?" in cmd


class TestBRollPreview:
    """Test B-roll preview generation."""

    @pytest.fixture
    def service(self) -> BRollIntegrationService:
        """Create a BRollIntegrationService instance."""
        return BRollIntegrationService()

    @pytest.mark.asyncio
    async def test_get_broll_preview_frame_full_replace(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test preview frame generation for full replace mode."""
        preview_content = b"PNG image content"
        preview_file = tmp_path / "preview.png"

        with patch.object(service, "_run_ffmpeg", new_callable=AsyncMock), \
             patch("tempfile.NamedTemporaryFile") as mock_temp:

            # Mock temp file
            mock_temp.return_value.__enter__.return_value.name = str(preview_file)
            preview_file.write_bytes(preview_content)

            result = await service.get_broll_preview_frame(
                main_video_path="/main.mp4",
                broll_path="/broll.mp4",
                timestamp=5.0,
                mode="full_replace",
            )

        assert result == preview_content

    @pytest.mark.asyncio
    async def test_get_broll_preview_frame_pip_overlay(
        self, service: BRollIntegrationService, tmp_path: Path
    ):
        """Test preview frame generation for PIP overlay mode."""
        with patch.object(service, "_get_video_info", new_callable=AsyncMock) as mock_info, \
             patch.object(service, "_run_ffmpeg", new_callable=AsyncMock) as mock_ffmpeg, \
             patch("tempfile.NamedTemporaryFile") as mock_temp:

            mock_info.return_value = {"width": 1920, "height": 1080}

            preview_file = tmp_path / "preview.png"
            preview_file.write_bytes(b"PNG content")
            mock_temp.return_value.__enter__.return_value.name = str(preview_file)

            await service.get_broll_preview_frame(
                main_video_path="/main.mp4",
                broll_path="/broll.mp4",
                timestamp=5.0,
                mode="pip_overlay",
            )

            # Should include overlay filter for PIP
            cmd = mock_ffmpeg.call_args[0][0]
            assert "overlay" in str(cmd) or "-filter_complex" in cmd


class TestServiceSingleton:
    """Test service singleton pattern."""

    def test_get_broll_integration_service(self):
        """Test singleton service getter."""
        service1 = get_broll_integration_service()
        service2 = get_broll_integration_service()

        assert service1 is service2
        assert isinstance(service1, BRollIntegrationService)
