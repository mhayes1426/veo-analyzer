# Veo Analyzer

Veo Analyzer is a local-first companion to [Veo Backup](https://github.com/mhayes1426/veo-backup). It adds chapter-style basketball highlights to full-game recordings without modifying, deleting, or re-encoding the original MP4 files.

Veo Analyzer begins with a trustworthy manual review and labeling interface, then uses locally trained models to propose human-reviewed made-basket candidates. Other basketball event types remain future work.

## Safety contract

- Veo Backup remains an independent application and is not modified by this project.
- The Backup media directory is mounted read-only in the Analyzer container.
- Analyzer owns a separate SQLite database and never writes to Veo Backup's database.
- A recording must be registered in Analyzer before it can be streamed.
- Resolved media paths must remain beneath the configured media root.
- Original MP4 files are immutable. Sidecars and optional derived containers are written to a separate export directory.
- Model runs may replace their own unreviewed candidates, but never approved, rejected, corrected, or manual events.

## Proposed Unraid mounts

| Host path | Container path | Mode | Purpose |
|---|---|---:|---|
| Your basketball media folder | `/data` | read-only | Existing recordings |
| Your Analyzer app-data folder | `/config` | read-write | Analyzer database, settings, logs, model cache |
| Your Analyzer export folder | `/exports` | read-write | JSON, FFmetadata, and optional derived MKV files |

Veo Analyzer discovers recordings directly from the media folder and does not require a Veo Backup configuration mount.

## Documentation

- [Architecture and ownership](docs/architecture.md)
- [Phased implementation plan](docs/implementation-plan.md)

## Run locally with Docker Desktop

1. Create `sample-media` in this project and place or link one test MP4 inside it. The folder is mounted read-only and ignored by Git.
2. Start the application:

   ```bash
   docker compose -f compose.local.yml up --build
   ```

3. Open [http://localhost:8090](http://localhost:8090).

The local Analyzer database is stored in `local-data/config`; JSON chapter exports are stored in `local-data/exports`. Both directories are ignored by Git. Stop the container with:

```bash
docker compose -f compose.local.yml down
```

To use a folder elsewhere on the computer, replace `./sample-media` on the left side of the `/data:ro` mapping in `compose.local.yml`. Docker Desktop must be allowed to share that folder.

The image supports `PUID` and `PGID` for writable bind mounts. The Unraid template defaults to Unraid's standard `99:100`; the local Compose file defaults to `1000:1000`.

## Container variants

- `ghcr.io/mhayes1426/veo-analyzer:latest` — lightweight CPU/UI image.
- `ghcr.io/mhayes1426/veo-analyzer:latest-gpu` — Unraid/NVIDIA image with CUDA-enabled PyTorch. This image is several gigabytes larger.

The GPU image still requires the host NVIDIA driver and container runtime. Open **Training & GPU** in the dashboard to verify NVIDIA, PyTorch, CUDA, VRAM, and annotation readiness.

## Training export privacy

The YOLO dataset export contains reviewed frame images, normalized basketball/rim labels, opaque recording/event/frame IDs, sequence outcomes, and source fingerprints. It excludes recording titles, media paths, LAN addresses, credentials, and sessions. Dataset splits are assigned by entire recording to prevent frames from the same game appearing across training and evaluation sets.

The NVIDIA image also provides local Training Jobs on the Training & GPU page. A job snapshots reviewed frames into recording-level train/validation/test splits, trains an Ultralytics YOLO11 detector on CUDA, records progress and metrics in Analyzer's SQLite database, and stores versioned model artifacts under `/config/training/jobs`. Training is explicitly started by the user and never modifies source videos.

## Made-basket analysis

After a model finishes training, open **Analyze** to run it against an adjustable section of a cataloged game. Calibration jobs sample a short range, display timestamped basketball/hoop detections, and explain whether each ball position is above, near, or below the rim. Full-game jobs apply the same detector in one-minute batches and look for a horizontally aligned above-to-below rim crossing.

Crossings become `made_basket` candidates with model/run provenance, confidence, and the configured pre/post playback window. They are never auto-approved. The game player provides one-click approval/rejection plus timestamp correction, re-analysis never overwrites reviewed decisions, and long-running jobs can be stopped from the UI. Job state, normalized detections, evidence previews, and logs remain local under `/config/analysis/jobs`; source recordings remain read-only.

## Status

Manual chapters, annotation, local GPU training, explainable calibration, and human-reviewed made-basket candidate creation are implemented.
