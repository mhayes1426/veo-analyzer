# Architecture and ownership

## Container boundary

Veo Analyzer is one separately deployable container with four internal layers:

1. **Web application** — LAN-only UI and authenticated API, video streaming, review workflow, and exports.
2. **Application services** — recording catalog, event lifecycle, job orchestration, and immutable-decision rules.
3. **Workers** — FFprobe inspection, export jobs, and eventually CPU/GPU inference.
4. **Adapters** — local filesystem discovery and optional read-only Veo Backup metadata import.

A single image keeps Unraid installation simple. Web and worker processes may be split into separate containers later while continuing to share only Analyzer's `/config` and `/exports` volumes. The inference implementation remains behind a worker interface so CPU and NVIDIA runtimes can use the same job and event contracts.

## Repository layout

```text
veo-analyzer/
├── app/
│   ├── api/                 # HTTP routes, range streaming, request schemas
│   ├── core/                # configuration, security, logging
│   ├── db/                  # models, migrations, repositories
│   ├── domain/              # recording, event, analysis-job rules
│   ├── importers/           # filesystem and read-only Backup adapters
│   ├── services/            # catalog, review, export, orchestration
│   ├── static/              # browser assets
│   ├── templates/           # server-rendered LAN UI
│   └── workers/             # probe/export/inference jobs
├── migrations/              # Analyzer-only SQLite migrations
├── models/                  # model manifests; weights excluded from Git
├── tests/
│   ├── integration/
│   ├── security/
│   └── unit/
├── docs/
├── docker/
├── Dockerfile
└── compose.example.yml
```

Recommended initial stack: Python 3.12, FastAPI/Starlette, SQLAlchemy 2, Alembic, SQLite in WAL mode, server-rendered templates with small progressive-enhancement JavaScript, FFmpeg/FFprobe, and pytest. This fits the existing operational environment while keeping the browser and database contracts replaceable.

## Database ownership

Analyzer exclusively owns `/config/analyzer.db`. Only Analyzer migrations touch it. Veo Backup exclusively owns its database.

### `recordings`

| Field | Notes |
|---|---|
| `recording_id` | Analyzer UUID; stable public identity |
| `source_kind` | `filesystem` or `veo_backup` |
| `source_recording_key` | Nullable opaque external identity; never assumed to be a foreign key |
| `media_path` | Path relative to `/data`, never an arbitrary absolute path |
| `title`, `recorded_at` | Local display metadata |
| `duration_seconds`, `size_bytes`, `content_fingerprint` | Probe and identity checks |
| `availability` | `present`, `missing`, `changed`, `unsupported` |
| `created_at`, `updated_at`, `last_seen_at` | UTC timestamps |

`media_path` is unique. A quick fingerprint should combine size and sampled content; a full hash is optional because Backup already verifies its files and full-game hashing is expensive.

### `events`

| Field | Notes |
|---|---|
| `event_id` | UUID |
| `recording_id` | Analyzer-owned foreign key |
| `event_type` | Initially `made_basket`, plus controlled future values |
| `event_time_seconds` | Canonical moment in the recording |
| `play_from_seconds`, `play_until_seconds` | Validated playback window |
| `confidence` | Nullable 0–1 model confidence |
| `review_status` | `candidate`, `approved`, `rejected` |
| `source` | `manual`, `model`, `corrected` |
| `model_version` | Nullable immutable model identifier |
| `analysis_job_id` | Nullable provenance link |
| `supersedes_event_id` | Nullable correction lineage |
| `created_at`, `updated_at`, `reviewed_at` | UTC timestamps |

Constraints enforce `0 <= play_from <= event_time <= play_until <= duration`, when duration is known. Manual additions default to approved; model output always starts as candidate. Editing a model event records it as corrected and retains provenance.

### `analysis_jobs`

Stores `job_id`, `recording_id`, `job_type`, `status`, requested model/version and parameters, progress, timestamps, error summary, and a cancellation flag. Only one active job of a given type may exist per recording. Jobs are durable and resumable after container restart.

### `model_runs` and decision preservation

Each inference execution records its exact model version, parameter manifest, code version, input fingerprint, and metrics. Before a re-run, Analyzer may remove or supersede only events satisfying all of these conditions:

- `source = model`
- `review_status = candidate`
- produced by the prior run being replaced

Approved, rejected, manual, and corrected rows are immutable to automated replacement. Rejected decisions remain available as negative labels.

### Audit and settings

`event_revisions` records before/after event values and the local actor/session. `settings` stores schema-versioned non-secret configuration. Secrets, if later required, belong in environment variables or a separate protected file, never sidecars or exported datasets.

## Safest Veo Backup integration

Integration preference, safest first:

1. **Filesystem catalog**: scan `/data/Veo` read-only, accept supported regular media files, probe them, and register relative paths. This has no dependency on Backup internals.
2. **Supported metadata snapshot**: consume a versioned JSON export from Veo Backup if one becomes available later.
3. **Read-only SQLite adapter**: optionally mount `/backup-config` read-only, open a copied snapshot using SQLite immutable/read-only mode, and map only explicitly supported schema versions. Never query the live file through a long-lived connection.

The adapter is an anti-corruption layer: external columns are converted into Analyzer records and no Backup table, row ID, lock, migration, or Python module becomes part of Analyzer's domain model. Unsupported schemas fail closed and fall back to filesystem discovery.

## Secure media streaming

The browser requests `GET /api/recordings/{recording_id}/media`; it never supplies a filesystem path.

For each request the server:

1. Loads the known recording row.
2. Joins its stored relative path to the configured media root.
3. Resolves symlinks and verifies the resolved path remains beneath the resolved media root.
4. Requires a regular file with an allowed media type and checks its current identity against the catalog.
5. Parses exactly one RFC-compatible byte range, rejects malformed or excessive ranges, and returns `206`, `Content-Range`, `Accept-Ranges`, and the exact content length.
6. Uses bounded streaming chunks and does not expose directory listings or absolute paths in responses or logs.

The UI should be LAN-bound by default, but “LAN-only” is not an authentication mechanism. Phase 1 includes a local login, secure session cookies, CSRF protection for mutations, rate limits for login and streaming abuse, and configurable trusted proxy handling.

## Portable exports

JSON sidecars are written under `/exports/<recording-id>/chapters.json`, not beside the read-only MP4. The versioned document contains a recording fingerprint, non-sensitive display metadata, time base, event provenance, review state, and playback windows. Writes use a temporary file plus atomic rename.

FFmetadata export escapes reserved characters and emits approved chapters by default. Chapter timestamps use millisecond time base. An optional MKV job maps the original streams with `-c copy` and imports chapters; output is always a new file under `/exports`. It verifies source and destination identities and never replaces the MP4. Container incompatibility fails visibly rather than silently re-encoding.

## Deployment shape

```yaml
services:
  veo-analyzer:
    image: ghcr.io/mhayes1426/veo-analyzer:<pinned-version>
    ports:
      - "<lan-port>:8080"
    volumes:
      - /mnt/user/media/basketball:/data:ro
      - /mnt/user/appdata/veo-analyzer:/config:rw
      - /mnt/user/media/basketball-analysis:/exports:rw
      # Optional metadata adapter only:
      # Optional metadata adapters should use a separate read-only mount.
    environment:
      - MEDIA_ROOT=/data/Veo
      - ANALYZER_DB=/config/analyzer.db
      - EXPORT_ROOT=/exports
    restart: unless-stopped
```

GPU support later adds the NVIDIA runtime/device reservation without changing volumes or database ownership. CPU mode remains valid, though slower.
