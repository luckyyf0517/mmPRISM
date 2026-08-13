# CE-CNSL Intake And Pose Pilot

Status: current
Owner: Data rebuild lane
Authority scope: Receipt, metadata repair, label preservation, and bounded pose-pilot gates for CE-CNSL.
Last reviewed: 2026-08-13

## Purpose And Priority

CE-CNSL is a P1 follow-on public source. Under `DEC-054`, this operation is paused while CSL-Daily remains the P0
stable baseline. The page preserves the already reviewed procedure; it is not an instruction to start intake now.

## Activation Gate

Do not download the source, implement an adapter, select/run the pilot, or allocate GPU capacity until both conditions
are recorded:

1. CSL-Daily has an accepted `annotation_v2 -> synthetic FMCW -> OmniHand -> pose-only WaveLLM` stable loop with
   frozen run/evaluation evidence.
2. The coordinator explicitly reactivates `add-ce-cnsl-parallel-source` after reviewing what the CSL-Daily
   implementation established about shared and dataset-specific interfaces.

There is no automatic activation based only on elapsed time or free compute.

This workspace owns CE-CNSL intake and processed delivery. No separate CE-CNSL workspace is created. Shared code
belongs under `src/mmprism/`; dataset-specific filename, label, and source-layout behavior belongs in a thin adapter.

## Dataset Boundary

- Stable ID: `DATASET-CE-CNSL`.
- Preserve `gloss_raw`, versioned `gloss_normalized`, `regional_note`, and `spoken_chinese` separately.
- Keep official train/dev/test assignments for benchmark reproduction.
- Treat the official split as non-participant-disjoint: all 12 signer labels occur in every split.
- Do not call CE-CNSL `CSL-Daily-v2`, silently concatenate datasets, or publish one mixed aggregate score.
- Visual background diversity is not evidence of radar multipath diversity after skeleton-only simulation.

## Gate 1: Immutable Source Receipt

After activation, download the complete archive into a versioned `incoming/` or `external/` destination configured
through the data root. Before extraction or processing, record:

- source URL, retrieval time, archive byte count, SHA-256, and complete extracted inventory;
- repository commit and exact train/dev/test CSV checksums;
- codec, FPS, resolution, duration, and decodable frame count for every video;
- one-to-one sample-number coverage between videos and CSV rows;
- explicit upstream license/permission status for processing and derivative redistribution.

Absence of an explicit license does not prevent a private feasibility pilot, but it blocks public redistribution of
the source or derived pose/synthetic-radar corpus until clarified.

## Gate 2: Label And Signer Audit

The published old labels have an acknowledged signer mismatch from H onward. Build sample identity from the stable
sample number, not the directory signer name alone. Freeze a repair table that records published CSV signer,
directory signer, repaired signer, evidence, and review status. Manually inspect boundary samples for every signer.

The label adapter must be lossless. Normalization is a named transform with its own version and may not overwrite
gesture variants, subject/object annotations, direction annotations, or regional notes. Build formal vocabularies
from the configured training split only; using dev/test to construct a vocabulary is prohibited.

## Gate 3: Stratified Pose Pilot

Select 120--240 sequences before full processing. The frozen selection covers:

- all 12 repaired signers and each phone/device group;
- 720p, 1080p, and any other observed frame geometry/FPS;
- short, median, and long sequence strata;
- indoor/outdoor, difficult lighting, hand-face overlap, motion blur, and small-hand framing;
- regional-note and CSL-Daily-overlap/new-vocabulary examples.

Run the same output contract and QC semantics as CSL-Daily `annotation_v2`: preserve native pose/scores; emit finite
canonical `[T,2,24,3]`, confidence, joint validity, imputation, and frame mask; quarantine decode/inference failures;
and preserve aspect ratio rather than stretching frames to `256x256`.

The pilot report includes exact coverage, failure categories, hand/joint validity and confidence distributions,
temporal jump/bone-length/left-right alerts, plus deterministic source/overlay review. It compares strata rather than
reporting only one average success rate.

## Promotion And Scheduling

Full CE-CNSL annotation is permitted only after all three gates pass and an explicit acceptance record identifies
the frozen source, repaired labels, pilot manifest, config/model fingerprints, and QC result. Promotion reuses the
generic scheduler, annotation payload, QC, simulation, and delivery functions developed for CSL-Daily; it does not
copy or import a `csl_daily` business wrapper whose assumptions are dataset-specific.

After activation, GPU scheduling must preserve CSL-Daily and revision-critical capacity. Free compute by itself does
not authorize CE-CNSL execution.

After promotion, the first semantic experiment is sequential `CSL-Daily -> CE-CNSL` adaptation. CE-CNSL-only and
balanced joint/rehearsal runs are comparisons. Every report evaluates CSL-Daily and CE-CNSL separately to measure
new-domain gain and catastrophic forgetting.

## Verification

```bash
uv run python scripts/audit_docs.py
uv run pytest tests/unit/test_document_governance.py
openspec validate add-ce-cnsl-parallel-source --strict --no-interactive
git diff --check
```
