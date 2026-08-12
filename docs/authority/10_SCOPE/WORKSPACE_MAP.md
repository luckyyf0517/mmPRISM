# Workspace Map

Status: current
Owner: mmPRISM coordinator
Authority scope: Business workspace boundaries, logical ownership, and cross-workspace flow.
Last reviewed: 2026-08-12

## Business Boundaries

| Workspace | Owns | Receives | Produces |
|---|---|---|---|
| `csl_news_annotation` | CSL-News source gates, RTMW3D annotation, QC, source/pose manifests | Official RGB/labels and pinned pose model | Frozen visual pose+caption manifest and validation |
| `data_rebuild` | Data intake, radar recovery, split, quarantine, model-ready delivery | Frozen sources, acquisition/calibration evidence | Task-specific manifests, splits, delivery inventories |
| `sign_language_collection` | Revision-focused video-guided CSL collection and basic session QC | Frozen CSL videos/text, minimal consent, fixed radar/reference setup | Raw sessions, participant-type metadata, compact QC, frozen manifest |
| `omnihand_training` | Pose reconstruction train/resume/evaluate | Radar cube, metric pose, split | Checkpoint, pose predictions, pose metrics |
| `wavellm_training` | Sign-language generation train/resume/evaluate | Pose/confidence/radar feature/caption, split | Adapter, text predictions, language metrics |
| `paper_revision` | Reviewer closure, evidence promotion, manuscript and response | Frozen producer evidence | Evidence map, manuscript changes, response letter |

`sign_language_collection` owns only new acquisition and session-level acceptance. `data_rebuild` owns
processing, canonical split construction and task-specific delivery after accepting a frozen collection manifest.
The legacy self-collected cohort is non-semantic gesture evidence and does not count toward the new semantic
cohort or translation evidence.

## Shared Code Boundary

Workspaces are execution and ownership views, not Python packages. The following remain project-level:

```text
src/mmprism/contracts/
src/mmprism/config/
src/mmprism/runtime/
src/mmprism/artifacts/
src/mmprism/assets/
src/mmprism/evaluation/
src/mmprism/release/
src/mmprism/cli.py
configs/
tests/
```

Workspace indexes link to the root-owned modules, configuration, scripts, and tests they use. Physical
movement of these paths or changes to public interfaces require a separate OpenSpec change.

## Handoff Rule

Routine same-workspace handoff is limited to task, state, result, evidence, next action, and blocker. A
cross-workspace data or paper-evidence delivery additionally identifies the producer commit, immutable
location, manifest or inventory hash, and validation status. Existing manifests and run receipts carry
these fields whenever possible; no duplicate handoff report is required.
