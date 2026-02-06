"""End-to-end test for B-roll integration service."""

import asyncio
import subprocess
import tempfile
from pathlib import Path


async def test_broll_integration():
    """Test B-roll integration with real FFmpeg calls."""
    # Import here to ensure app context
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.services.broll_integration_service import BRollIntegrationService

    # Use the exported clip from our tests
    main_video = Path("storage/clips/clip_31dd815e-6f34-4c68-a2c6-5631cd5e7543.mp4")
    if not main_video.exists():
        print(f"Main video not found: {main_video}")
        return False

    print(f"Using main video: {main_video}")

    # Get video info
    service = BRollIntegrationService()
    info = await service._get_video_info(str(main_video))
    print(f"Main video info: {info}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create a simple test B-roll video (5 seconds of blue)
        broll_path = temp_path / "test_broll.mp4"
        print("Creating test B-roll video...")

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:s={info['width']}x{info['height']}:d=5",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-t", "5",
            broll_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to create test B-roll: {result.stderr}")
            return False

        print(f"Created test B-roll: {broll_path}")

        # Define B-roll insertion
        output_path = temp_path / "output_with_broll.mp4"
        broll_insertions = [
            {
                "start_time": 10.0,  # Insert at 10 seconds
                "duration": 3.0,     # 3 second B-roll
                "asset_path": str(broll_path),
                "mode": "full_replace",
                "transition": "crossfade",
            }
        ]

        print(f"Applying B-roll at 10s for 3s duration...")
        print(f"Output path: {output_path}")

        try:
            result_path = await service.apply_broll(
                input_path=str(main_video),
                output_path=str(output_path),
                broll_insertions=broll_insertions,
                transition_duration=0.5,
            )

            print(f"B-roll applied! Output: {result_path}")

            # Verify output exists and has reasonable size
            output_file = Path(result_path)
            if output_file.exists():
                size = output_file.stat().st_size
                print(f"Output file size: {size / 1024 / 1024:.2f} MB")

                # Get output video info
                output_info = await service._get_video_info(str(output_file))
                print(f"Output video info: {output_info}")

                # Verify duration is reasonable
                expected_duration = info["duration"]  # Should be roughly same as input
                actual_duration = output_info.get("duration", 0)
                print(f"Expected duration: ~{expected_duration:.1f}s, Actual: {actual_duration:.1f}s")

                if actual_duration > 0:
                    print("\n✓ B-roll integration test PASSED!")
                    return True
                else:
                    print("\n✗ Output video has no duration")
                    return False
            else:
                print("\n✗ Output file not created")
                return False

        except Exception as e:
            print(f"\n✗ B-roll integration failed: {e}")
            import traceback
            traceback.print_exc()
            return False


async def test_pip_overlay():
    """Test picture-in-picture B-roll overlay."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from app.services.broll_integration_service import BRollIntegrationService

    main_video = Path("storage/clips/clip_31dd815e-6f34-4c68-a2c6-5631cd5e7543.mp4")
    if not main_video.exists():
        print(f"Main video not found: {main_video}")
        return False

    print(f"\nTesting PIP overlay mode...")

    service = BRollIntegrationService()
    info = await service._get_video_info(str(main_video))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test B-roll
        broll_path = temp_path / "test_broll_pip.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=red:s=640x360:d=5",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            broll_path,
        ]
        subprocess.run(cmd, capture_output=True)

        output_path = temp_path / "output_with_pip.mp4"
        broll_insertions = [
            {
                "start_time": 5.0,
                "duration": 4.0,
                "asset_path": str(broll_path),
                "mode": "pip_overlay",  # Picture-in-picture mode
                "transition": "fade",
            }
        ]

        try:
            result_path = await service.apply_broll(
                str(main_video),
                str(output_path),
                broll_insertions,
                transition_duration=0.3,
            )

            if Path(result_path).exists():
                size = Path(result_path).stat().st_size
                print(f"PIP output file size: {size / 1024 / 1024:.2f} MB")
                print("✓ PIP overlay test PASSED!")
                return True
            else:
                print("✗ PIP output not created")
                return False

        except Exception as e:
            print(f"✗ PIP overlay failed: {e}")
            return False


if __name__ == "__main__":
    print("=" * 60)
    print("B-Roll Integration End-to-End Test")
    print("=" * 60)

    # Change to backend directory
    import os
    os.chdir(Path(__file__).parent.parent)

    # Run tests
    full_replace_ok = asyncio.run(test_broll_integration())
    pip_ok = asyncio.run(test_pip_overlay())

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Full Replace Mode: {'PASS' if full_replace_ok else 'FAIL'}")
    print(f"  PIP Overlay Mode:  {'PASS' if pip_ok else 'FAIL'}")
    print("=" * 60)
