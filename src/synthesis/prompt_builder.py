"""
Multimodal forensic prompt construction for Large Vision-Language Models (VLMs).
"""

from __future__ import annotations

from typing import Any
import pixeltable as pxt


def format_forensic_prompt(
    incident_query: str,
    visual_hits: list[dict[str, Any]],
    audio_hits: list[dict[str, Any]],
    osha_hits: list[dict[str, Any]],
) -> str:
    """
    Constructs an evidentiary forensic prompt fusing visual keyframe metadata,
    synchronized acoustic Whisper transcripts, and retrieved OSHA regulations.
    """
    v_str = "\n".join(
        f"- [t={float(h.get('timestamp_sec', 0.0)):.1f}s | Zone: {h.get('camera_zone')} "
        f"| Primary Obj: {h.get('detected_object')} | CLIP Sim: {float(h.get('visual_score', 0.0)):.3f}] "
        f"Source: {h.get('video_url')}"
        for h in (visual_hits or [])
    ) or "- No keyframes exceeded visual similarity threshold."

    a_str = "\n".join(
        f"- [{float(h.get('start_sec', 0.0)):.1f}s - {float(h.get('end_sec', 0.0)):.1f}s "
        f"| Acoustic Sim: {float(h.get('audio_score', 0.0)):.3f}] \"{str(h.get('transcript', '')).strip()}\""
        for h in (audio_hits or [])
    ) or "- No vocal radio callouts detected in acoustic stream."

    r_str = "\n".join(
        f"- [{h.get('regulation')}] (Rel: {float(h.get('score', 0.0)):.3f}): {h.get('passage')}"
        for h in (osha_hits or [])
    ) or "- Standard general duty clause (OSHA Section 5(a)(1))."

    return (
        f"INCIDENT INVESTIGATION QUERY: {incident_query}\n\n"
        f"1. VISUAL KEYFRAME EVIDENCE (CLIP ViT-B/32 + DETR Object Detection):\n{v_str}\n\n"
        f"2. SYNCHRONIZED AUDIO/RADIO TRANSCRIPTS (Silence-Snapped Whisper ASR):\n{a_str}\n\n"
        f"3. APPLICABLE OSHA REGULATORY STANDARDS:\n{r_str}\n\n"
        "Synthesize an official OSHA Form 301 & Near-Miss Forensic Triage Report containing:\n"
        "  (A) Exact temporal window (start/end seconds) and camera URI;\n"
        "  (B) Corroborated visual + acoustic root cause analysis;\n"
        "  (C) Specific OSHA 29 CFR 1910 paragraph violated and financial/compliance risk;\n"
        "  (D) Immediate engineering and administrative corrective actions."
    )


# Expose as Pixeltable UDF for declarative TableModel execution
build_forensic_prompt = pxt.udf(format_forensic_prompt)
