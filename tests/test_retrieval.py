"""
Unit tests for cross-modal retrieval and OSHA grounding.
"""

from src.retrieval.engine import execute_hybrid_retrieval
from src.retrieval.osha_matcher import match_osha_standard_offline


def test_hybrid_retrieval_dock_bay():
    """Tests cross-modal retrieval on dock_bay_4 forklift leak scenario."""
    query = "forklift skidding near chemical drums with hydraulic fluid leak"
    res = execute_hybrid_retrieval(query, target_zone="dock_bay_4")

    assert "visual_hits" in res
    assert "audio_hits" in res
    assert "osha_hits" in res
    assert len(res["visual_hits"]) > 0
    assert len(res["audio_hits"]) > 0
    assert len(res["osha_hits"]) > 0

    top_v = res["visual_hits"][0]
    assert top_v["camera_zone"] == "dock_bay_4"
    assert top_v["visual_score"] > 0.20
    assert "video_url" in top_v

    top_a = res["audio_hits"][0]
    assert "hydraulic" in top_a["transcript"].lower() or "reach truck" in top_a["transcript"].lower()


def test_hybrid_retrieval_chemical_storage():
    """Tests cross-modal retrieval on chemical_storage thermal ignition scenario."""
    query = "sparks and smoke plume rising near solvent tank"
    res = execute_hybrid_retrieval(query, target_zone="chemical_storage")

    top_v = res["visual_hits"][0]
    assert top_v["camera_zone"] == "chemical_storage"
    assert top_v["visual_score"] > 0.25


def test_osha_matcher_citations():
    """Verifies that OSHA matcher correctly identifies specific CFR 1910 standards."""
    forklift_rules = match_osha_standard_offline("reach truck hydraulic leak")
    assert any("1910.178" in r["regulation"] for r in forklift_rules)

    chemical_rules = match_osha_standard_offline("flammable solvent vapor ignition")
    assert any("1910.106" in r["regulation"] for r in chemical_rules)

    loto_rules = match_osha_standard_offline("technician conveyor lockout bypass")
    assert any("1910.147" in r["regulation"] for r in loto_rules)

