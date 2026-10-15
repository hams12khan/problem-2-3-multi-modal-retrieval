"""
Unit tests for multimodal prompt construction and OSHA Form 301 report synthesis.
"""

from src.synthesis.prompt_builder import format_forensic_prompt
from src.synthesis.forensic_reporter import (
    synthesize_offline_report,
    generate_forensic_report,
)


def test_format_forensic_prompt():
    """Tests structured prompt assembly across visual, audio, and OSHA passages."""
    prompt = format_forensic_prompt(
        incident_query="forklift leak",
        visual_hits=[{"timestamp_sec": 18.0, "camera_zone": "dock_bay_4", "detected_object": "forklift", "visual_score": 0.35, "video_url": "s3://test.mp4"}],
        audio_hits=[{"start_sec": 14.0, "end_sec": 29.0, "audio_score": 0.68, "transcript": "Radio check"}],
        osha_hits=[{"regulation": "OSHA 1910.178", "passage": "Truck maintenance", "score": 0.85}],
    )

    assert "INCIDENT INVESTIGATION QUERY: forklift leak" in prompt
    assert "VISUAL KEYFRAME EVIDENCE" in prompt
    assert "SYNCHRONIZED AUDIO/RADIO TRANSCRIPTS" in prompt
    assert "OSHA 1910.178" in prompt


def test_synthesize_offline_report_structure():
    """Verifies that the generated OSHA Form 301 triage report contains mandatory compliance sections."""
    report = synthesize_offline_report(
        incident_query="reach truck leak",
        target_zone="dock_bay_4",
        visual_hits=[{"timestamp_sec": 18.0, "visual_score": 0.342, "detected_object": "forklift"}],
        audio_hits=[{"start_sec": 14.2, "end_sec": 29.8, "transcript": "Hydraulic line burst"}],
        osha_hits=[{"regulation": "OSHA 29 CFR 1910.178(p)(1)", "passage": "Defective trucks taken out of service"}],
    )

    assert "=== OSHA FORM 301 & INCIDENT FORENSIC TRIAGE REPORT ===" in report
    assert "Temporal Localization Window" in report
    assert "Grounded Regulatory Citation: OSHA 29 CFR 1910.178(p)(1)" in report
    assert "$42,000" in report
    assert "Corrective Actions:" in report


def test_generate_forensic_report_fallback():
    """Verifies deterministic fallback works when OPENAI_API_KEY is not set."""
    report = generate_forensic_report(
        incident_query="test query",
        target_zone="assembly_line_2",
        visual_hits=[],
        audio_hits=[],
        osha_hits=[],
    )
    assert "OSHA FORM 301" in report
    assert "assembly_line_2" in report

