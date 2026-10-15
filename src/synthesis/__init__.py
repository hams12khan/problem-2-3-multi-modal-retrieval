"""
Forensic report synthesis and multimodal prompt generation.
"""

from src.synthesis.prompt_builder import build_forensic_prompt
from src.synthesis.forensic_reporter import (
    local_forensic_synthesizer,
    generate_forensic_report,
)

__all__ = [
    "build_forensic_prompt",
    "local_forensic_synthesizer",
    "generate_forensic_report",
]
