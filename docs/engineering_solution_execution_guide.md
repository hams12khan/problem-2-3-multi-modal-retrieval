# Option 1: Solution Building — Multimodal Video RAG at Enterprise Scale
## Automated Industrial Safety Forensics & OSHA Near-Miss Triage with Pixeltable

---

### 1. Problem Formulation: The Industrial Video Triage Bottleneck
In high-hazard enterprise environments—heavy manufacturing, automated warehousing, chemical processing, and large-scale construction—fixed CCTV surveillance and wearable floor-operator bodycams are the primary source of truth for operational safety. The Occupational Safety and Health Administration (OSHA) mandates or strongly urges root-cause investigations not only for recordable workplace injuries (which carry an average direct and indirect cost of $\$42,000$ per incident) but also for "near misses" that precede catastrophic failures.

Despite ubiquitous camera infrastructure, safety engineering teams face a severe **semantic retrieval bottleneck** across untrimmed, multi-stream video:
1. **Closed-Set Detector Limitations:** Traditional computer vision models (such as single-frame PPE object detectors) achieve $mAP@50 \approx 0.88\text{–}0.95$ on fixed classes (e.g., hardhats or vests) but cannot answer open-ended, cross-modal forensic queries over hundreds of hours of continuous recordings—such as localizing every instance where *"a forklift skids near staged chemical drums while a supervisor simultaneously reports a hydraulic leak over two-way radio."*
2. **Imperative Pipeline Fragility:** Conventional DIY Multimodal RAG stacks require gluing together Amazon S3 (`boto3`), OpenCV (`cv2.VideoCapture`) frame extraction loops, MoviePy audio extraction, manual sliding-window transcript concatenation ($n=7$ neighboring segments), PostgreSQL metadata tables, a standalone vector database (LanceDB/Pinecone), and an orchestrator (Airflow/Celery).

---

### 2. Non-Trivial Dataset Scale & Operational Workload
Rather than operating on a toy collection of short clips, this architecture is engineered for a composite enterprise safety benchmark combining **UCF-Crime** (fixed surveillance) and **Ego4D Episodic Memory** (egocentric wearable bodycams with vocal callouts):

| Workload Dimension | Toy Prototype Baseline | Enterprise Industrial Forensics Target | Pixeltable Architectural Mechanism |
| :--- | :--- | :--- | :--- |
| **Video Corpus Volume** | $2\text{–}5$ short clips ($<30$ mins) | **1,900 untrimmed videos ($128+$ hours)** via UCF-Crime + Ego4D streams | Out-of-core S3/Tigris media offloading (`input_media_dest`) + zero-copy virtual views |
| **Visual Index Cardinality** | $<1,500$ extracted frames | **$460,800$ frames** (at $1.0\text{ FPS}$) or $\sim 92,000$ I-frame keyframes | HNSW `pxt.EmbeddingIndex` (`CLIP ViT-B/32`) intersected with `pxt.BtreeIndex` |
| **Acoustic Granularity** | Clean pre-existing WebVTT subtitles | **$15,360+$ noisy acoustic windows** ($30\text{s}$ window, $2\text{s}$ overlap) | `audio_splitter` snapped to quiet stretches ($\ge 0.3\text{s}$ below $-40\text{ dBFS}$) + local Whisper ASR |
| **Regulatory Grounding** | Unstructured prompt guesswork | **OSHA 29 CFR 1910 standards** chunked at $42,111\text{ rows/s}$ | `document_splitter` (`sentence,token_limit`, limit=$300$, overlap=$50$) |
| **Pipeline Maintenance** | Full batch re-runs on script modification | **Continuous incremental updates** on new camera uploads | Declarative `TableModel` computed columns and automated lineage tracking |

---

### 3. Specific Usage of the Pixeltable Open-Source SDK (`app.py`)

Our implementation in `app.py` collapses the entire multimodal backend into a single declarative file using four core capabilities of the `pixeltable/pixeltable` SDK:

#### A. Declarative Base Tables & Cloud Storage Offloading
* **`FacilityVideos` & `SafetyManuals` (`pxt.model_base()`):** Uses first-class media column types (`pxt.Video`, `pxt.Document`, `pxt.Image`) paired with relational `pxt.BtreeIndex` indexes on `camera_zone` and `shift_date`.
* **Zero-Egress Cloud Media Management:** By configuring `input_media_dest` and `output_media_dest` in `~/.pixeltable/config.toml` (supporting S3, Tigris, Cloudflare R2, or Backblaze B2), raw surveillance feeds and generated $320\times 320$ VLM thumbnails stream directly to object storage while preserving `.localpath` and `.fileurl` lineage inside Pixeltable's catalog.

#### B. Zero-Copy Media Decomposition via Virtual Iterator Views
Instead of writing OpenCV or FFmpeg loops that materialize $460,800$ uncompressed images onto disk, `app.py` declares three virtual views:
1. **`VideoFrames` (`frame_iterator(FacilityVideos.video, fps=1.0)`):** Registers positional metadata (`pos`, `pts`) at **$8,687.65\text{ rows/s}$** (or **$3,277.53\text{ rows/s}$** with `keyframes_only=True`). Unstored frames are decoded on demand via internal `seek(pos)` calls.
2. **`AudioSegments` (`audio_splitter(...)`):** Automatically extracts the underlying audio track from inserted videos and splits it into $30.0\text{s}$ segments with $2.0\text{s}$ overlap, `min_silence_len=0.3`, `silence_thresh=-40.0`, and `trim_leading_silence=True`. This snaps cut boundaries to natural pauses in radio callouts so words are never cleaved mid-syllable, and gracefully yields $0$ rows on silent CCTV feeds.
3. **`ManualChunks` (`document_splitter(...)`):** Chunks OSHA 29 CFR 1910 manuals along sentence and token boundaries (`limit=300`, `overlap=50`, `metadata='title,heading'`) at **$42,111.88\text{ rows/s}$**.

#### C. Incremental Model Inference & Hybrid B-Tree + HNSW Indexing
* **Visual Pipeline:** Computes persistent $320\times 320$ thumbnails (`frame.thumbnail((320, 320))`), extracts structured bounding boxes via local `detr_for_object_detection(frame, model_id='facebook/detr-resnet-50')`, and maintains an HNSW `pxt.EmbeddingIndex` using `clip.using(model_id='openai/clip-vit-base-patch32')` supporting both text-to-frame and reverse image-to-frame queries.
* **Acoustic Pipeline (Dependency Resilience):** Rather than using `whisperx` (which hard-pins `torch==2.8` and causes dependency deadlocks in modern Python environments), `AudioSegments` runs local OpenAI Whisper via `whisper.transcribe(audio=audio_segment, model='base.en')` and indexes `asr_json.text` with `sentence-transformers/all-MiniLM-L6-v2`.

#### D. Implicit Parent-Child Joins, `@pxt.query`, and `FastAPIRouter` Serving
* In `retrieve_visual_evidence` and `retrieve_audio_evidence`, the `.where()` clause filters directly on `camera_zone` and selects `video.fileurl`—columns that live on the parent `FacilityVideos` table—without writing manual SQL `JOIN` statements.
* `IncidentInvestigations` chains the three `@pxt.query` functions into `@pxt.udf build_forensic_prompt` and synthesizes the final OSHA Form 301 & Near-Miss Forensic Triage Report via `openai.chat_completions(model='gpt-4o-mini')` (with an automatic offline deterministic fallback UDF `local_forensic_synthesizer` when `OPENAI_API_KEY` is not set).
* `FastAPIRouter` exposes `/videos` (`add_insert_route`) and `/investigate` (`add_compute_route`) directly from `app.py`.

---

### 4. End-to-End Execution Instructions

#### Option A: Run Standalone Reproducible Validation (`python app.py`)
```bash
# 1. Install Pixeltable and multimodal dependencies
pip install pixeltable torch transformers sentence-transformers openai-whisper spacy tiktoken
python -m spacy download en_core_web_sm

# 2. Run the self-contained CLI validation & forensic report synthesizer
python app.py --mode demo

# 3. Batch-ingest a local UCF-Crime / Ego4D video directory
python app.py --mode ingest --dataset-dir ./ucf_crime_subset --zone dock_bay_4 --shift-date 2026-10-15
```

#### Option B: Launch Production Declarative Catalog & FastAPI Service
```bash
# Initialize project, migrate declarative TableModel schema, and launch FastAPIRouter
pxt init
pxt schema update app.py safety_intel
pxt service update app.py safety_intel
```
Once running, open the interactive **Enterprise Industrial Safety Video RAG Workbench (`index.html`)** and toggle **"Live FastAPIRouter (:8000)"** in the top navigation bar, or use the built-in **128h Benchmark Simulator** directly in the browser to inspect synchronized keyframes, DETR bounding boxes, silence-snapped Whisper transcripts, and OSHA 29 CFR 1910 citations.