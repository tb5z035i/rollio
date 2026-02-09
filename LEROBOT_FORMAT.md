# LeRobot Dataset Format (v2.1 vs v3.0)

This document gives a **general, practical description** of the on-disk data format used by **LeRobot** datasets, covering both the **v2.1 (episode-based)** and **v3.0 (sharded / file-based)** layouts.

> Core idea (both versions):  
> - **Tabular / low-dimensional signals** are stored in **Parquet**.  
> - **Visual streams** (RGB cameras, sometimes others) are stored as **MP4** video streams.  
> - A **canonical schema** and file path templates live in `meta/info.json`.

---

## 1) Common concepts (v2.1 and v3.0)

### 1.1 Canonical schema: `meta/info.json`
`meta/info.json` is the contract describing the dataset:

- `codebase_version`: dataset format version (`v2.1`, `v3.0`, …)
- `fps`: nominal sampling rate used to interpret `timestamp` / frame stepping
- `total_episodes`, `total_frames` (or equivalent)
- **Path templates** for locating tabular and video data (exact template fields differ between v2.1 and v3.0)
- `features`: dictionary describing every key present in samples:
  - **dtype** (e.g., `float32`, `int64`, `bool`, `video`)
  - **shape** (e.g., action dimension, state dimension)
  - optional `names` for dimensions (e.g., joint names)
  - for visual features: video metadata such as codec / pixel format / resolution / fps

**Implication:** the schema is global for the dataset. If you mix different robots (different joint counts), you typically:
- use separate datasets, or
- define a unified state/action interface (padding/masks + fixed ordering), because `shape` is fixed per key.

### 1.2 Modalities and typical key naming
Common key families (exact names vary by dataset but follow the same pattern):

- **Robot signals (tabular; Parquet)**
  - `observation.state` (vector)
  - `action` (vector)
  - `timestamp` (scalar, seconds from episode start)
  - indexing: `episode_index`, `frame_index`, `index`
  - optional: `task_index`, `next.done`, rewards, etc.

- **Visual streams (video; MP4)**
  - `observation.images.<camera_name>` (multiple cameras are multiple streams)
  - occasionally other streams (e.g., segmentation / depth), but depth often needs special handling for fidelity.

### 1.3 Alignment and sampling
LeRobot treats the tabular `timestamp` as the canonical time axis:
- For a given sample time `t`, it decodes the closest video frame(s) in each stream near `t`
- Multi-camera alignment is achieved by querying each stream using the same `timestamp` (and any offsets such as history windows)

### 1.4 Space saving / encoding
- **Videos:** stored as MP4 with a chosen codec (often modern codecs for size efficiency).  
- **Tabular:** Parquet uses columnar storage and compression.  
- **Depth:** lossy video codecs can harm depth fidelity; many pipelines keep depth lossless (e.g., uint16 images or lossless codecs) if metric quality matters.

---

## 2) LeRobot v2.1 — episode-based storage

### 2.1 High-level idea
In **v2.1**, **each episode is stored separately**:

- One **Parquet file per episode** (tabular)
- Typically one **MP4 per camera per episode** (visual)

### 2.2 Typical directory layout (conceptual)
```text
dataset_root/
├─ meta/
│  ├─ info.json
│  ├─ stats.json
│  ├─ episodes.jsonl / episodes.*        (episode list/metadata; varies)
│  └─ tasks.jsonl / tasks.*              (optional)
├─ data/
│  └─ chunk-XYZ/
│     ├─ episode_000000.parquet
│     ├─ episode_000001.parquet
│     └─ ...
└─ videos/
   └─ chunk-XYZ/
      └─ {video_key}/
         ├─ episode_000000.mp4
         ├─ episode_000001.mp4
         └─ ...
```

**How you find files:** via templates stored in `meta/info.json`, often in the form:
- `data_path`: `data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet`
- `video_path`: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`

### 2.3 What is inside an episode Parquet
The episode Parquet contains a row per timestep:

- `timestamp`
- `observation.state` (vector)
- `action` (vector)
- indices: `frame_index`, `episode_index`, `index`
- optional task / done flags

### 2.4 Multi-camera (RGB) in v2.1
Multiple cameras are represented as **multiple video keys**:
- each `video_key` corresponds to a separate MP4 file **per episode**
- file paths are constructed by the `video_path` template + the `video_key`

### 2.5 Depth (if present) in v2.1
Depth is sometimes stored as a visual stream, but due to compression issues, many users store depth losslessly (implementation depends on dataset; check `meta/info.json` and any dataset-specific notes).

---

## 3) LeRobot v3.0 — sharded / file-based storage

### 3.1 High-level idea
In **v3.0**, the storage unit is no longer “one episode per file.” Instead:

- Many episodes are concatenated into **Parquet shards**
- Many episodes are concatenated into **MP4 shards per camera**
- Episode boundaries are recovered via **episode metadata** under `meta/episodes/`

### 3.2 Typical directory layout (conceptual)
```text
dataset_root/
├─ meta/
│  ├─ info.json
│  ├─ stats.json
│  ├─ tasks.*                         (parquet/jsonl; dataset dependent)
│  └─ episodes/
│     └─ chunk-XYZ/
│        └─ file-ABC.parquet          (episode boundary/lookup metadata)
├─ data/
│  └─ chunk-XYZ/
│     └─ file-ABC.parquet             (tabular rows for many episodes)
└─ videos/
   └─ {video_key}/
      └─ chunk-XYZ/
         └─ file-ABC.mp4              (frames for many episodes)
```

### 3.3 How you locate the correct rows/frames for an episode
v3 introduces **relational episode metadata** (in `meta/episodes/...parquet`), which tells the loader:
- which shard file contains a given episode
- the row ranges (Parquet) / frame ranges (MP4) that belong to that episode

### 3.4 Templates in `meta/info.json` (typical v3.0)
Common patterns look like:
- `data_path`: `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`
- `video_path`: `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4`

### 3.5 Multi-camera (RGB) in v3.0
Each camera stream is typically stored under:
- `videos/<video_key>/chunk-.../file-....mp4`

All camera streams are queried using the same canonical `timestamp` (plus any requested offsets).

---

## 4) Quick comparison

| Aspect | v2.1 | v3.0 |
|---|---|---|
| Storage unit | **Episode** | **Shard** (many episodes per file) |
| Tabular data | `episode_XXXX.parquet` | `file-XXX.parquet` (many episodes) |
| Visual data | `episode_XXXX.mp4` per camera | `file-XXX.mp4` per camera (many episodes) |
| Episode boundaries | implied by filenames | stored in `meta/episodes/*.parquet` |
| Goal | simplicity | scalability + fewer files + streaming |

---

## 5) What to check for your specific dataset

1. Open `meta/info.json`:
   - list of `features` (keys)
   - which keys are `dtype: video` (cameras)
   - shapes for `observation.state` and `action` (and any other robot signals)

2. Inspect `meta/episodes/*` (v3.0 only):
   - confirms how episode-to-shard mapping and offsets are represented

3. Confirm depth handling:
   - if depth is stored as video, verify precision and artifacts
   - if depth must be metric-accurate, prefer lossless storage

---

*End of document.*
