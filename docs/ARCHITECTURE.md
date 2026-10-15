# Technical Architecture: Declarative Multimodal Video RAG
## Deep-Dive into Pixeltable Data Infrastructure, Virtual Views, and Industrial Safety Forensics

---

### 1. Architectural Paradigm: Eliminating Multimodal Data Plumbing

Building a high-throughput video RAG system traditionally requires stitching together a complex, multi-system imperative stack:

```
[Traditional DIY Stack]
Raw Video Files ──► OpenCV cv2.VideoCapture (Frame Extraction Loop to Disk)
                 ──► MoviePy (MP3 Audio Extraction)
                 ──► Whisper (Transcript Slices) ──► Custom Sliding Window Concatenator (n=7)
                 ──► Boto3 (S3 Sync)
                 ──► PostgreSQL (Relational Metadata)
                 ──► LanceDB / Pinecone (Vector Index)
                 ──► Airflow / Celery (Retry & Batch Scheduling Glue)
```

**The Pain Points of the Imperative Stack:**
1. **Massive Storage Amplification:** Materializing 1 frame per second across 128 hours yields $460,800$ uncompressed JPEG/PNG files ($>180\text{ GB}$ of local disk usage).
2. **Audio Cleaving:** Fixed-interval audio chunking (e.g. cutting every 30s regardless of speech) slices words in half, corrupting transcript accuracy.
3. **Dual-Database Synchronization Lag:** Keeping relational camera metadata (e.g. `camera_zone = 'dock_bay_4'`) synchronized with an external vector database requires custom transactional logic.
4. **Pipeline Fragility:** Corrupt MP4 headers or codec drops crash the entire batch job.

**The Pixeltable Solution:**
Pixeltable replaces this entire multi-system infrastructure with a single declarative Python application:

```
[Pixeltable Declarative Architecture]
Base Tables (FacilityVideos, SafetyManuals)
     │
     ├──► Virtual View (VideoFrames): Zero-copy seek(pos) decoding at 8,687.65 rows/s
     │         └──► Computed Column: DETR Object Detection
     │         └──► HNSW Index: CLIP ViT-B/32 (Text & Image reverse search)
     │
     ├──► Virtual View (AudioSegments): audio_splitter snapped to pauses >= 0.3s
     │         └──► Computed Column: Local Whisper base.en transcription
     │         └──► HNSW Index: all-MiniLM-L6-v2 semantic search
     │
     └──► Virtual View (ManualChunks): document_splitter for OSHA 29 CFR 1910
               └──► HNSW Index: all-MiniLM-L6-v2 regulatory vector grounding
```

---

### 2. Zero-Copy Media Decomposition via Virtual Iterator Views

The core innovation that allows SentinelPXT to scale to 128+ hours is Pixeltable's **Virtual View Layer**:
* **On-Demand Decoding (`frame_iterator`):** When videos are inserted into `FacilityVideos`, `VideoFrames` records only lightweight metadata (`pos`, `pts`) without writing images to disk. During query execution, Pixeltable's C-accelerated decoder seeks directly to the target frame index via `seek(pos)` and decodes only the requested keyframes.
* **Silence-Aware Acoustic Windowing (`audio_splitter`):**
  ```python
  iterator = audio_splitter(
      audio=FacilityVideos.video,
      duration=30.0,
      overlap=2.0,
      min_silence_len=0.3,
      silence_thresh=-40.0,
      trim_leading_silence=True,
  )
  ```
  This configuration snaps segment cut boundaries to natural pauses ($\ge 0.3\text{s}$ below $-40\text{ dBFS}$) in radio callouts. Words are never cleaved mid-syllable, and silent CCTV feeds gracefully yield 0 rows without crashing.

---

### 3. Unified Hybrid Query Execution Plan

Unlike conventional stacks that execute a vector query in Pinecone followed by a SQL query in PostgreSQL and an in-memory intersection in Python, Pixeltable's single-pass planner intersects relational predicates and vector similarities:

```sql
-- Conceptual Unified Execution Plan inside Pixeltable
SELECT 
    v.fileurl AS video_url,
    f.camera_zone,
    f.pts AS timestamp_sec,
    f.primary_object,
    SIMILARITY(f.frame, :query_text) AS visual_score
FROM safety_intel.video_frames f
JOIN safety_intel.facility_videos v ON f.video_id = v.video_id
WHERE f.camera_zone = :target_zone
  AND SIMILARITY(f.frame, :query_text) > 0.20
ORDER BY visual_score DESC
LIMIT 5;
```

Thanks to **implicit parent-child joins**, queries written in Python select parent table columns (`frames_t.video.fileurl`) and filter on `frames_t.camera_zone` directly on the child view without manual `JOIN` boilerplate.

---

### 4. Dependency Resilience: Whisper vs WhisperX

While Pixeltable includes an optional `whisperx` module with speaker diarization, an analysis of `pyproject.toml` reveals that `whisperx` hard-pins `torch==2.8` (causing dependency deadlocks in modern Python environments).

SentinelPXT solves this by standardizing on **native local OpenAI Whisper (`whisper.transcribe(model='base.en')`)**:
1. Eliminates torch version deadlocks.
2. Runs 100% locally on CPU or CUDA without external API keys or subscription fees.
3. Pairs with `sentence-transformers/all-MiniLM-L6-v2` for sub-millisecond transcript vector search.

---

### 5. Closed-Loop Machine Learning Refinement

When daily RAG investigations surface novel near-miss frames, engineers can continuously retrain downstream object detectors or fine-tune embedding spaces:
* **PyTorch Streaming Export:**
  ```python
  dataset = frames_view.to_pytorch_dataset('pt')
  ```
* **COCO Annotation Export:**
  ```python
  from src.utils.export import export_coco_annotations
  export_coco_annotations(detections, "annotations/near_miss_coco.json")
  ```

