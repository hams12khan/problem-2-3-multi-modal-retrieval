"""
Unit tests for silence-snapped acoustic slicing and synthetic media generation.
"""

import tempfile
import wave
from pathlib import Path
from src.utils.synthetic_data import generate_synthetic_safety_assets


def test_synthetic_safety_assets_generation():
    """Tests that synthetic assets are generated with proper audio sample rate and silence duration."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        assets = generate_synthetic_safety_assets(tmp_path)

        assert assets["osha_doc"].exists()
        assert assets["sample_wav"].exists()
        assert assets["sample_img"].exists()

        # Validate WAV properties
        with wave.open(str(assets["sample_wav"]), "r") as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getsampwidth() == 2
            duration = wav_file.getnframes() / 16000
            assert duration >= 5.0

        # Validate OSHA text content
        content = assets["osha_doc"].read_text(encoding="utf-8")
        assert "OSHA 29 CFR 1910.178" in content
        assert "Lockout/Tagout" in content

