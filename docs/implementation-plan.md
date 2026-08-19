# Phased implementation plan

## Phase 0 — Foundation and safety contract

**Goal:** a reproducible, testable container skeleton with explicit invariants.

Deliverables:

- Python application, dependency lock, multi-stage Docker image, non-root runtime, health endpoint, and example Unraid/Compose configuration.
- Typed configuration for media, config, and export roots; startup refuses unsafe overlap between media and export roots.
- Analyzer-owned SQLite setup, WAL/busy-timeout policy, migrations, structured logs with private paths and titles redacted by default.
- Unit/security test harness and CI for lint, types, migrations, and container smoke test.
- Documented backup procedure using SQLite's online backup API.

Exit criteria: container starts without Backup present; all writable data remains under `/config` or `/exports`; attempts to write below `/data` fail in deployment and tests.

## Phase 1 — Manual chapters MVP

**Goal:** users can safely catalog, watch, label, and review a full game.

Deliverables:

- Filesystem discovery and FFprobe catalog with stable recording fingerprints.
- Recording, event, revision, and job schemas with migrations and integrity constraints.
- Local login, secure sessions, CSRF protection for mutations, and configurable trusted-proxy handling.
- Record-ID-only Range streaming endpoint with traversal, symlink-escape, malformed-range, stale-file, and authorization tests.
- Game page with full-game HTML video, current timestamp, “Add Highlight Here,” event type, editable pre/post timing, approve/reject/delete, chapter list, and previous/next highlight controls.
- Keyboard shortcuts that avoid conflicting with text inputs and include an on-screen help panel.
- Defaults: event at playhead, play from 8 seconds before through 5 seconds after, clamped to media bounds; configurable globally and editable per event.
- Versioned JSON sidecar and approved-only FFmetadata export with deterministic ordering and atomic writes.

Exit criteria: a user can label an entire game without touching the original file; reloading preserves player position and review state; Chrome/Firefox/Safari seeking works; security tests cover every path boundary.

## Phase 2 — Operational hardening and label export

**Goal:** make manual labels dependable enough to become training truth.

Deliverables:

- Audit history, schema-compatible import/export, and Analyzer SQLite backups.
- Durable job runner, cancellation, restart recovery, progress display, and cross-process job locks.
- Catalog reconciliation for missing/changed recordings without deleting their events.
- Training-data export for approved events, corrected candidates, rejected candidates, and manually added missed events; manifests contain fingerprints and relative identifiers, not private paths or titles.
- Optional read-only Veo Backup snapshot adapter, gated by detected/supported schema version.
- On-demand chapter-enabled MKV remux with stream copy, verification, cancellation, and free-space checks.

Exit criteria: restore tests recover database plus exports; a changed or missing source cannot silently attach labels to the wrong video; export datasets are reproducible and privacy-reviewed.

## Phase 3 — Made-basket candidate pipeline

**Goal:** models propose made-basket chapters for human review.

Deliverables:

- Pluggable inference worker with CPU and NVIDIA profiles and sampled-frame caching under `/config`.
- Representative local annotation workflow for ball, hoop, and player detection.
- Detector plus temporal scoring: ball approaches rim, crosses the rim region above-to-below, persists across frames, and optionally agrees with a scoreboard delta.
- Candidate deduplication, confidence calibration, model/run provenance, and decision-preserving re-analysis.
- Review queue sorted for efficient confirmation and correction.
- Held-out-game evaluation: precision, recall, event timestamp error, and playback-boundary error.

Release gate: at least 90% precision, 80% recall, and under 2 seconds median/declared target timestamp error on games excluded from training. Metrics must be reported by game as well as aggregate; the model does not auto-approve.

## Phase 4 — Made-basket quality and active learning

**Goal:** improve robustness across gyms, lighting, zoom, occlusion, and scoreboard layouts.

Deliverables:

- Error taxonomy and slice metrics.
- Hard-negative mining from rejected candidates and hard-positive sampling from manual missed events.
- Model registry with immutable versions, compatibility manifest, rollback, and reproducible evaluation.
- Optional team/direction/scoreboard modules that degrade gracefully when unavailable.

Exit criteria: a newly promoted model beats the current model on the frozen held-out set and does not regress critical slices beyond an agreed tolerance.

## Phase 5 — Additional basketball events

Add event types independently, with separate evaluation gates:

1. Blocks and rebounds using ball/hoop/player tracks around shot attempts.
2. Steals using team/player tracking and possession transitions, excluding rebounds, inbounds, jump balls, dead balls, and loose-ball recoveries.
3. Other configurable highlights only after their label definition and evaluation protocol are written.

No event type is promoted solely on anecdotal results. Every model remains candidate-only until its measured precision and the review workflow justify a policy change.

## Initial implementation slices

The first engineering work should land as small, independently testable slices:

1. Container skeleton, configuration, database migration, and health check.
2. Read-only catalog plus recording detail API.
3. Secure Range streaming with adversarial tests.
4. Manual event CRUD, invariants, and audit revisions.
5. Player and keyboard navigation.
6. JSON and FFmetadata exports.
7. Operational backup/recovery and label export.

## Explicit non-goals for the MVP

- No automated AI detection.
- No physical clip generation by default.
- No modification, relocation, or deletion of original media.
- No direct writes, migrations, locks, or runtime imports involving Veo Backup.
- No cloud upload, telemetry, or public exposure by default.
- No promise that MP4 can hold the desired chapters portably; MKV is the derived, stream-copy container option.

## Decisions to validate during Phase 0

- Exact browser support and authentication preference for the LAN dashboard.
- Whether `/data` contains only completed, verified games or Analyzer must consume an explicit completion signal.
- Whether the optional Backup metadata adapter is valuable enough to justify version coupling.
- Expected GPU model/VRAM and acceptable CPU processing time.
- Whether one recording can contain multiple games or periods requiring a richer game/segment model.
