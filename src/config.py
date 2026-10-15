"""
Configuration and settings for SentinelPXT Multimodal Video RAG.
"""

from __future__ import annotations

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DOCS_DIR = PROJECT_ROOT / "docs"
UI_DIR = SRC_DIR / "ui"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "ucf_crime_subset"
DEFAULT_TEMP_DIR = Path(os.environ.get("SENTINEL_TEMP_DIR", Path.home() / ".sentinel_pxt"))

# Namespace
DEFAULT_NAMESPACE = "safety_intel"
DEMO_NAMESPACE = "safety_intel_demo"

# Retrieval & Threshold Defaults
DEFAULT_VISUAL_SIM_THRESHOLD = 0.20
DEFAULT_AUDIO_SIM_THRESHOLD = 0.25
DEFAULT_TOP_K_VISUAL = 5
DEFAULT_TOP_K_AUDIO = 5
DEFAULT_TOP_K_OSHA = 3

# Pretrained Model Identifiers
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DETR_MODEL_ID = "facebook/detr-resnet-50"
TEXT_EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL = "base.en"
VLM_MODEL = "gpt-4o-mini"

# Zero-Copy Audio Slicing Parameters (snaps to >=0.3s silence stretches)
AUDIO_DURATION = 30.0
AUDIO_OVERLAP = 2.0
AUDIO_MIN_SILENCE_LEN = 0.3
AUDIO_SILENCE_THRESH = -40.0
AUDIO_TRIM_LEADING_SILENCE = True

# OSHA Regulatory Document Chunker Parameters
DOC_LIMIT = 300
DOC_OVERLAP = 50
DOC_SEPARATORS = "sentence,token_limit"

# Facility Zones for B-tree Relational Indexing
SUPPORTED_ZONES = [
    "dock_bay_4",
    "chemical_storage",
    "bodycam_shift_2",
    "assembly_line_2",
]

# API & Cloud Storage Endpoints
API_HOST = os.environ.get("SENTINEL_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("SENTINEL_API_PORT", 8000))
S3_INPUT_MEDIA_DEST = os.environ.get("PIXELTABLE_INPUT_MEDIA_DEST", "s3://industrial-safety-vault/production/uploads/")
S3_OUTPUT_MEDIA_DEST = os.environ.get("PIXELTABLE_OUTPUT_MEDIA_DEST", "s3://industrial-safety-vault/production/generated/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
