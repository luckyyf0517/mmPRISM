# Project Authority Changelog

Status: current
Owner: mmPRISM coordinator
Authority scope: Material changes to project boundaries, shared contracts, or project Authority.
Last reviewed: 2026-08-12

## 2026-08-12

- Replaced the monolithic revision control plane with project Authority and five business workspaces.
- Kept canonical shared code, configuration, and tests at repository root.
- Established lightweight routine handoff and frozen cross-workspace delivery rules.
- Preserved old documentation paths as compatibility entrypoints and moved dated evidence to Logs.
- Accepted `DEC-039`, superseding `DEC-001` as the current documentation control-plane decision.
- Added a dedicated semantic sign-language collection workspace with a roughly 30-participant target, while
  explicitly excluding legacy non-semantic gestures from semantic cohort and translation evidence.
- Accepted Chinese Sign Language (CSL) as the primary language of the new semantic collection while leaving its
  precise variety/register and written translation target for expert review.
- Simplified the collection around the revision deadline: seek 3--4 professional/proficient CSL contributors when
  available and scale with video-guided volunteers, without maintaining a recruitment funnel.
- Archived CSL-News as checkpoint-side visual-pose evidence and removed its local source/download/cache layer;
  CSL-Daily intake and the new semantic CSL collection are now the active data-rebuild paths.
- Registered the author's incoming historical WaveLLM bundle as preservation-only while transfer is in progress.
  The existing CSL-News-derived mT5-only export remains a fallback until a stable receipt and controlled audit
  establish the incoming checkpoint identity and admissible role.
- Added a lightweight, non-authoritative literature-note area and recorded Uni-Sign's pre-training-scale evidence
  without promoting its full-data result into a revision requirement.
- Relocated the original-submission forensic codebase (root `run_*.py`, `config/`, legacy `src/` modules, and
  legacy shell wrappers) into an explicit read-only `legacy/` directory without content changes (`DEC-047`);
  `src/` now contains only the canonical `mmprism` package.
- Aligned the CSL-Daily controlled reproduction plan across data, reconstruction, translation, and paper workspaces:
  immutable baseline/candidate annotation, explicit pose-only semantics, checkpoint-bound cross-fitted features, and
  a strict separation between synthetic control, historical replay, and reviewer-facing real-radar evidence.
- Added one cross-workspace research execution model defining cam-pose, synthetic radar, synthetic/real-domain
  mmw-pose, radar features, model roles, stage handoffs, and the CSL-Daily end-to-end control path; workspace
  operation pages retain their own commands, gates, outputs, and current state.
- Implemented the first CSL-Daily WaveLLM control slice: v2 JSONL+NPY manifests and formal train/evaluate now bind
  `pose_only` or `pose_plus_radar_feature`, with a parameter-free pose-only path and checkpoint mode enforcement.
  Final Parquet pose-only delivery now omits radar features at the schema/row/reader level; CubeNet
  prediction/feature provenance remains the explicit follow-up contract.

## 2026-08-13

- Paused local CSL-Daily execution for compute-server migration: the `annotation_v2` queue remains initialized but
  paused, and the two pending batch benchmark jobs were cancelled before starting. Added a target-server acceptance
  and handoff runbook that keeps GFS data and historical checkpoints immutable while requiring a fresh CUDA 12.8
  environment, core-model, distributed, and RTMW3D runtime gates before work resumes.
- Recorded that the historical WaveLLM bundle is now staged under `log/archived/` but remains preservation-only;
  receipt, format/world-size, metadata/tensor, and controlled-load audit are separate pending work.

- Accepted the then-provisional `DEC-048` pose-only ordering. `DEC-051` subsequently recorded the v1 pilot's
  material contract failure and made full-corpus `annotation_v2` mandatory; checkpoint-bound feature/fusion remains
  a non-blocking third-stage comparison.
- Accepted `DEC-049`: CSL-Daily persists pre-beamforming synthetic FMCW and derives CubeNet power cubes at runtime.
  The existing direct-cube materialization path is now an engineering prototype, not a formal CSL-Daily product.
- Accepted `DEC-050`: the retained CSL-Daily source is the expanded JPEG tree plus official metadata and review
  videos. The duplicate full archive and its transfer splits are removed after an inventory record. Since no
  historical full pose/signal/feature products arrived, a versioned, quality-controlled full-corpus cam-pose
  reconstruction is a P0 prerequisite for synthetic radar, OmniHand, and WaveLLM work.
- Accepted `DEC-051`: the 54-sample RTMW3D annotation v1 pilot is frozen as diagnostic evidence, not a training
  source, because completed arrays retain hand NaNs and its sidecars omit native pose/scores and validity data.
  Contract-complete annotation v2 becomes the required full-corpus cam-pose reconstruction before simulation.
- Accepted `DEC-052`: CE-CNSL becomes a P1 parallel public source for vocabulary and heterogeneous-domain
  expansion while CSL-Daily remains the P0 baseline. Intake, signer/label audit, and a bounded pose pilot proceed
  in parallel; full processing is gated and cannot block CSL-Daily or new real-radar collection.
- Accepted `DEC-054`, superseding the execution schedule in `DEC-052`: preserve the CE-CNSL assessment and planned
  contracts, but pause all intake, adapter, pilot, and GPU work until the CSL-Daily end-to-end reproduction loop is
  accepted and the coordinator explicitly reactivates CE-CNSL.
- Accepted `DEC-056`: during CSL-Daily's late stable phase, explicit project-owner authorization may unlock CE-CNSL
  source download and immutable receipt only; all adapter, pilot, processing, and GPU work remains behind the stable
  loop and explicit-reactivation gate.
