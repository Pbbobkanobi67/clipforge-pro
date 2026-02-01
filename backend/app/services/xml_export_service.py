"""XML export service for Final Cut Pro and Premiere Pro."""

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET
from xml.dom import minidom

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _prettify_xml(elem: ET.Element) -> str:
    """Return a pretty-printed XML string."""
    rough_string = ET.tostring(elem, encoding="unicode")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def _seconds_to_fcpxml_time(seconds: float, fps: float = 30.0) -> str:
    """Convert seconds to FCPXML time format (frames/fps)."""
    frames = int(seconds * fps)
    # FCPXML uses rational time format: "frames/fps s"
    return f"{frames}/{int(fps)}s"


def _seconds_to_premiere_ticks(seconds: float) -> int:
    """Convert seconds to Premiere Pro ticks (254016000000 ticks per second)."""
    TICKS_PER_SECOND = 254016000000
    return int(seconds * TICKS_PER_SECOND)


class XMLExportService:
    """Service for exporting to NLE XML formats."""

    def __init__(self):
        self._export_dir = settings.storage_path / "exports"
        self._export_dir.mkdir(parents=True, exist_ok=True)

    async def export_fcpxml(
        self,
        clips: list[dict],
        video_path: str,
        output_path: str,
        project_name: str = "ClipForge Pro Export",
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        include_markers: bool = True,
        include_captions: bool = True,
        captions: Optional[list[dict]] = None,
    ) -> str:
        """
        Export clips to Final Cut Pro X XML format (FCPXML 1.10).

        Args:
            clips: List of clip data with start_time, end_time, title, transcript_excerpt
            video_path: Path to source video
            output_path: Output FCPXML file path
            project_name: Name of the project
            fps: Frame rate
            width: Video width
            height: Video height
            include_markers: Include clip markers
            include_captions: Include caption text
            captions: Optional caption segments

        Returns:
            Path to exported FCPXML file
        """
        # Create root element
        fcpxml = ET.Element("fcpxml", version="1.10")

        # Add resources
        resources = ET.SubElement(fcpxml, "resources")

        # Format resource
        format_id = "r1"
        format_elem = ET.SubElement(resources, "format",
                                     id=format_id,
                                     name=f"FFVideoFormat{height}p{int(fps)}",
                                     frameDuration=_seconds_to_fcpxml_time(1/fps, fps),
                                     width=str(width),
                                     height=str(height))

        # Asset resource (source video)
        asset_id = "r2"
        video_filename = Path(video_path).name
        asset = ET.SubElement(resources, "asset",
                              id=asset_id,
                              name=video_filename,
                              src=f"file://{video_path.replace(chr(92), '/')}")

        # Media reference
        media = ET.SubElement(asset, "media-rep",
                              kind="original-media",
                              src=f"file://{video_path.replace(chr(92), '/')}")

        # Create library
        library = ET.SubElement(fcpxml, "library")

        # Create event
        event = ET.SubElement(library, "event",
                              name=project_name,
                              uid=str(uuid.uuid4()).upper())

        # Create project
        project = ET.SubElement(event, "project",
                                name=project_name,
                                uid=str(uuid.uuid4()).upper())

        # Create sequence
        sequence = ET.SubElement(project, "sequence",
                                 format=format_id,
                                 tcStart="0s",
                                 tcFormat="NDF")

        # Add spine (main timeline)
        spine = ET.SubElement(sequence, "spine")

        timeline_offset = 0

        for i, clip in enumerate(clips):
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)
            duration = end_time - start_time
            title = clip.get("title", f"Clip {i+1}")

            # Add asset clip
            asset_clip = ET.SubElement(spine, "asset-clip",
                                       ref=asset_id,
                                       offset=_seconds_to_fcpxml_time(timeline_offset, fps),
                                       name=title,
                                       start=_seconds_to_fcpxml_time(start_time, fps),
                                       duration=_seconds_to_fcpxml_time(duration, fps),
                                       format=format_id)

            # Add marker if enabled
            if include_markers:
                marker = ET.SubElement(asset_clip, "marker",
                                       start=_seconds_to_fcpxml_time(0, fps),
                                       duration=_seconds_to_fcpxml_time(1/fps, fps),
                                       value=title)

                # Add note with transcript excerpt
                if clip.get("transcript_excerpt"):
                    note = ET.SubElement(marker, "note")
                    note.text = clip["transcript_excerpt"][:500]

            # Add captions as titles if enabled
            if include_captions and captions:
                clip_captions = [c for c in captions
                                 if start_time <= c.get("start_time", 0) <= end_time]

                for cap in clip_captions:
                    cap_start = cap["start_time"] - start_time
                    cap_end = cap["end_time"] - start_time
                    cap_duration = cap_end - cap_start

                    title_elem = ET.SubElement(asset_clip, "title",
                                               offset=_seconds_to_fcpxml_time(cap_start, fps),
                                               duration=_seconds_to_fcpxml_time(cap_duration, fps),
                                               name=cap.get("text", "")[:50])

                    # Title text
                    text_elem = ET.SubElement(title_elem, "text")
                    text_elem.text = cap.get("text", "")

            timeline_offset += duration

        # Write to file
        xml_string = _prettify_xml(fcpxml)

        # Add XML declaration
        xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_string

        Path(output_path).write_text(xml_string, encoding="utf-8")

        logger.info(f"Exported FCPXML: {output_path}")
        return output_path

    async def export_premiere_xml(
        self,
        clips: list[dict],
        video_path: str,
        output_path: str,
        project_name: str = "ClipForge Pro Export",
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        include_markers: bool = True,
        include_captions: bool = True,
        captions: Optional[list[dict]] = None,
    ) -> str:
        """
        Export clips to Adobe Premiere Pro XML format.

        Args:
            clips: List of clip data with start_time, end_time, title, transcript_excerpt
            video_path: Path to source video
            output_path: Output XML file path
            project_name: Name of the project
            fps: Frame rate
            width: Video width
            height: Video height
            include_markers: Include clip markers
            include_captions: Include caption text
            captions: Optional caption segments

        Returns:
            Path to exported XML file
        """
        # Create root element
        xmeml = ET.Element("xmeml", version="4")

        # Create sequence
        sequence = ET.SubElement(xmeml, "sequence")
        ET.SubElement(sequence, "name").text = project_name
        ET.SubElement(sequence, "uuid").text = str(uuid.uuid4())

        # Rate
        rate = ET.SubElement(sequence, "rate")
        ET.SubElement(rate, "timebase").text = str(int(fps))
        ET.SubElement(rate, "ntsc").text = "FALSE"

        # Duration (will be calculated)
        total_duration = sum(c.get("end_time", 0) - c.get("start_time", 0) for c in clips)
        total_frames = int(total_duration * fps)
        ET.SubElement(sequence, "duration").text = str(total_frames)

        # Timecode
        timecode = ET.SubElement(sequence, "timecode")
        ET.SubElement(timecode, "rate").text = str(int(fps))
        ET.SubElement(timecode, "string").text = "00:00:00:00"
        ET.SubElement(timecode, "frame").text = "0"
        ET.SubElement(timecode, "displayformat").text = "NDF"

        # Media
        media = ET.SubElement(sequence, "media")

        # Video track
        video_elem = ET.SubElement(media, "video")
        video_format = ET.SubElement(video_elem, "format")
        sample_characteristics = ET.SubElement(video_format, "samplecharacteristics")
        ET.SubElement(sample_characteristics, "width").text = str(width)
        ET.SubElement(sample_characteristics, "height").text = str(height)

        video_track = ET.SubElement(video_elem, "track")

        # Audio track
        audio_elem = ET.SubElement(media, "audio")
        audio_track = ET.SubElement(audio_elem, "track")

        timeline_frame = 0
        video_filename = Path(video_path).name

        for i, clip in enumerate(clips):
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)
            duration = end_time - start_time
            duration_frames = int(duration * fps)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            title = clip.get("title", f"Clip {i+1}")

            # Video clip item
            clipitem = ET.SubElement(video_track, "clipitem", id=f"clipitem-{i+1}")
            ET.SubElement(clipitem, "name").text = title

            ET.SubElement(clipitem, "enabled").text = "TRUE"
            ET.SubElement(clipitem, "duration").text = str(duration_frames)

            # Rate for clip
            clip_rate = ET.SubElement(clipitem, "rate")
            ET.SubElement(clip_rate, "timebase").text = str(int(fps))
            ET.SubElement(clip_rate, "ntsc").text = "FALSE"

            ET.SubElement(clipitem, "start").text = str(timeline_frame)
            ET.SubElement(clipitem, "end").text = str(timeline_frame + duration_frames)
            ET.SubElement(clipitem, "in").text = str(start_frame)
            ET.SubElement(clipitem, "out").text = str(end_frame)

            # File reference
            file_elem = ET.SubElement(clipitem, "file", id=f"file-{i+1}")
            ET.SubElement(file_elem, "name").text = video_filename
            ET.SubElement(file_elem, "pathurl").text = f"file://localhost/{video_path.replace(chr(92), '/')}"

            file_rate = ET.SubElement(file_elem, "rate")
            ET.SubElement(file_rate, "timebase").text = str(int(fps))
            ET.SubElement(file_rate, "ntsc").text = "FALSE"

            file_media = ET.SubElement(file_elem, "media")
            file_video = ET.SubElement(file_media, "video")
            file_audio = ET.SubElement(file_media, "audio")

            # Markers
            if include_markers:
                marker = ET.SubElement(clipitem, "marker")
                ET.SubElement(marker, "name").text = title
                ET.SubElement(marker, "in").text = "0"
                ET.SubElement(marker, "out").text = "-1"

                if clip.get("transcript_excerpt"):
                    comment = ET.SubElement(marker, "comment")
                    comment.text = clip["transcript_excerpt"][:500]

            # Audio clip item (linked)
            audio_clipitem = ET.SubElement(audio_track, "clipitem", id=f"audio-clipitem-{i+1}")
            ET.SubElement(audio_clipitem, "name").text = title
            ET.SubElement(audio_clipitem, "enabled").text = "TRUE"
            ET.SubElement(audio_clipitem, "duration").text = str(duration_frames)

            audio_clip_rate = ET.SubElement(audio_clipitem, "rate")
            ET.SubElement(audio_clip_rate, "timebase").text = str(int(fps))
            ET.SubElement(audio_clip_rate, "ntsc").text = "FALSE"

            ET.SubElement(audio_clipitem, "start").text = str(timeline_frame)
            ET.SubElement(audio_clipitem, "end").text = str(timeline_frame + duration_frames)
            ET.SubElement(audio_clipitem, "in").text = str(start_frame)
            ET.SubElement(audio_clipitem, "out").text = str(end_frame)

            # Link audio to video
            link = ET.SubElement(audio_clipitem, "link")
            ET.SubElement(link, "linkclipref").text = f"clipitem-{i+1}"

            timeline_frame += duration_frames

        # Write to file
        xml_string = _prettify_xml(xmeml)
        xml_string = '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE xmeml>\n' + xml_string

        Path(output_path).write_text(xml_string, encoding="utf-8")

        logger.info(f"Exported Premiere XML: {output_path}")
        return output_path

    async def export_edl(
        self,
        clips: list[dict],
        output_path: str,
        project_name: str = "ClipForge Pro Export",
        fps: float = 30.0,
    ) -> str:
        """
        Export clips to EDL (Edit Decision List) format.

        Simple text format compatible with most NLEs.

        Args:
            clips: List of clip data
            output_path: Output EDL file path
            project_name: Project name
            fps: Frame rate

        Returns:
            Path to exported EDL file
        """
        def _seconds_to_timecode(seconds: float, fps: float) -> str:
            """Convert seconds to SMPTE timecode."""
            frames = int(seconds * fps)
            ff = frames % int(fps)
            ss = (frames // int(fps)) % 60
            mm = (frames // (int(fps) * 60)) % 60
            hh = frames // (int(fps) * 3600)
            return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"

        lines = [
            f"TITLE: {project_name}",
            f"FCM: NON-DROP FRAME",
            "",
        ]

        timeline_seconds = 0

        for i, clip in enumerate(clips, 1):
            start_time = clip.get("start_time", 0)
            end_time = clip.get("end_time", 0)
            duration = end_time - start_time
            title = clip.get("title", f"Clip {i}")

            # Source in/out
            src_in = _seconds_to_timecode(start_time, fps)
            src_out = _seconds_to_timecode(end_time, fps)

            # Record in/out
            rec_in = _seconds_to_timecode(timeline_seconds, fps)
            rec_out = _seconds_to_timecode(timeline_seconds + duration, fps)

            # EDL entry format:
            # ### REEL_NAME   V C   SOURCE_IN SOURCE_OUT RECORD_IN RECORD_OUT
            lines.append(f"{i:03d}  AX       V     C        {src_in} {src_out} {rec_in} {rec_out}")
            lines.append(f"* FROM CLIP NAME: {title}")

            if clip.get("transcript_excerpt"):
                # Add as comment (limited length)
                excerpt = clip["transcript_excerpt"][:100].replace("\n", " ")
                lines.append(f"* COMMENT: {excerpt}")

            lines.append("")

            timeline_seconds += duration

        # Write to file
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")

        logger.info(f"Exported EDL: {output_path}")
        return output_path


# Singleton instance
_xml_export_service = None


def get_xml_export_service() -> XMLExportService:
    """Get singleton XML export service."""
    global _xml_export_service
    if _xml_export_service is None:
        _xml_export_service = XMLExportService()
    return _xml_export_service
