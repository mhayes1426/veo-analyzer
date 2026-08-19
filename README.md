# Veo Analyzer

Veo Analyzer is a local-first companion to [Veo Backup](https://github.com/mhayes1426/veo-backup). It adds chapter-style basketball highlights to full-game recordings without modifying, deleting, or re-encoding the original MP4 files.

The first release is intentionally manual: it provides a trustworthy review and labeling interface before any model is allowed to propose events. Later releases add human-reviewed made-basket detection, then other basketball events.

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

## Status

Architecture and milestone plan are defined. Application implementation begins with Phase 0 and Phase 1 in the plan.
