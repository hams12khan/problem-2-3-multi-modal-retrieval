"""
Zero-copy virtual iterator views for visual frame extraction, silence-snapped audio, and OSHA docs.
"""

from __future__ import annotations

from typing import Any
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import whisper
from pixeltable.functions.audio import audio_splitter
from pixeltable.functions.document import document_splitter
from pixeltable.functions.huggingface import clip, detr_for_object_detection, sentence_transformer
from pixeltable.functions.video import frame_iterator

from src.models.schema import TableModel, FacilityVideos, SafetyManuals
from src.config import (
    CLIP_MODEL_ID,
    DETR_MODEL_ID,
    TEXT_EMBED_MODEL_ID,
    WHISPER_MODEL,
    AUDIO_DURATION,
    AUDIO_OVERLAP,
    AUDIO_MIN_SILENCE_LEN,
    AUDIO_SILENCE_THRESH,
    AUDIO_TRIM_LEADING_SILENCE,
    DOC_LIMIT,
    DOC_OVERLAP,
    DOC_SEPARATORS,
)


class VideoFrames(
    TableModel,
    name="video_frames",
    base=FacilityVideos,
    iterator=frame_iterator(FacilityVideos.video, fps=1.0),
):
    """
    Zero-copy virtual iterator view extracting 1.0 FPS frames from FacilityVideos.
    Automatically exposes `frame` (pxt.Image), `pos` (int), and `pts` (float seconds),
    plus all parent columns (`camera_zone`, `shift_date`, `video`) without manual SQL JOINs.
    """
    thumb = pxt.Column(
        value=frame.thumbnail((320, 320)),  # type: ignore[name-defined]
    )
    detections = detr_for_object_detection(
        frame, model_id=DETR_MODEL_ID  # type: ignore[name-defined]
    )
    primary_object = detections.label_text[0]

    __indexes__ = [
        pxt.EmbeddingIndex(
            frame,  # type: ignore[name-defined]
            embedding=clip.using(model_id=CLIP_MODEL_ID),
            name="frame_clip_idx",
        )
    ]


class AudioSegments(
    TableModel,
    name="audio_segments",
    base=FacilityVideos,
    iterator=audio_splitter(
        audio=FacilityVideos.video,
        duration=AUDIO_DURATION,
        overlap=AUDIO_OVERLAP,
        min_silence_len=AUDIO_MIN_SILENCE_LEN,
        silence_thresh=AUDIO_SILENCE_THRESH,
        trim_leading_silence=AUDIO_TRIM_LEADING_SILENCE,
    ),
):
    """
    Silence-snapped acoustic iterator view (~30s windows, 2s overlap, snapped to >=0.3s
    quiet stretches below -40 dBFS). Runs OpenAI Whisper locally without external API keys.
    Gracefully yields 0 rows on silent CCTV streams.
    """
    asr_json = whisper.transcribe(
        audio=audio_segment, model=WHISPER_MODEL  # type: ignore[name-defined]
    )
    transcript = asr_json.text

    __indexes__ = [
        pxt.EmbeddingIndex(
            transcript,
            string_embed=sentence_transformer.using(
                model_id=TEXT_EMBED_MODEL_ID
            ),
            name="audio_transcript_idx",
        )
    ]


class ManualChunks(
    TableModel,
    name="manual_chunks",
    base=SafetyManuals,
    iterator=document_splitter(
        document=SafetyManuals.doc,
        separators=DOC_SEPARATORS,
        overlap=DOC_OVERLAP,
        limit=DOC_LIMIT,
        metadata="title,heading",
    ),
):
    """
    Sentence- and token-aware regulatory document chunker for OSHA standards grounding.
    """
    __indexes__ = [
        pxt.EmbeddingIndex(
            text,  # type: ignore[name-defined]
            string_embed=sentence_transformer.using(
                model_id=TEXT_EMBED_MODEL_ID
            ),
            name="osha_manual_idx",
        )
    ]


def initialize_imperative_workspace(namespace: str = "safety_intel_demo") -> dict[str, Any]:
    """
    Initializes the Pixeltable workspace imperatively using `pxt.create_dir`, `pxt.create_table`,
    and `pxt.create_view` for interactive CLI, Jupyter notebook, or test suite validation.
    """
    pxt.create_dir(namespace, if_exists="ignore")

    videos_t = pxt.create_table(
        f"{namespace}.facility_videos",
        {
            "video": pxt.Video,
            "camera_zone": pxt.String,
            "shift_date": pxt.String,
            "anomaly_tag": pxt.String,
        },
        if_exists="ignore",
    )

    manuals_t = pxt.create_table(
        f"{namespace}.safety_manuals",
        {
            "doc": pxt.Document,
            "regulation_code": pxt.String,
        },
        if_exists="ignore",
    )

    frames_v = pxt.create_view(
        f"{namespace}.video_frames",
        videos_t,
        iterator=frame_iterator(videos_t.video, fps=1.0),
        if_exists="ignore",
    )

    # Attach incremental CLIP visual-semantic embedding index on extracted frames
    frames_v.add_embedding_index(
        "frame",
        embedding=clip.using(model_id=CLIP_MODEL_ID),
        if_exists="ignore",
    )

    chunks_v = pxt.create_view(
        f"{namespace}.manual_chunks",
        manuals_t,
        iterator=document_splitter(
            document=manuals_t.doc,
            separators=DOC_SEPARATORS,
            overlap=DOC_OVERLAP,
            limit=DOC_LIMIT,
        ),
        if_exists="ignore",
    )

    chunks_v.add_embedding_index(
        "text",
        string_embed=sentence_transformer.using(
            model_id=TEXT_EMBED_MODEL_ID
        ),
        if_exists="ignore",
    )

    return {
        "videos": videos_t,
        "manuals": manuals_t,
        "frames": frames_v,
        "chunks": chunks_v,
    }
