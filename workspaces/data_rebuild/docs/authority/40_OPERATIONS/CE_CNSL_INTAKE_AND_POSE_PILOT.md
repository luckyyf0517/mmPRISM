# CE-CNSL Intake And Pose Pilot

Status: current
Owner: Data rebuild lane
Authority scope: Receipt, metadata repair, label preservation, and bounded pose-pilot gates for CE-CNSL.
Last reviewed: 2026-08-13

## Purpose And Priority

CE-CNSL is a P1 follow-on public source. Under `DEC-054` and `DEC-056`, implementation remains paused while
CSL-Daily is P0, with only an explicitly authorized late-phase archive transfer allowed before full activation. The
page preserves the reviewed procedure; it is not an instruction to start intake now.

## Authorization And Activation Gates

There are two distinct gates:

1. **Optional download authorization:** during CSL-Daily's late stable phase, the project owner may explicitly
   authorize CE-CNSL source download, checksum, and immutable receipt. Without that authorization, do not download.
   This authorization does not permit label repair, adapter work, pilot selection/execution, processing, or GPU use.
2. **Full activation:** implementation and compute work remain paused until both conditions are recorded:
   - CSL-Daily has an accepted `annotation_v2 -> synthetic FMCW -> OmniHand -> pose-only WaveLLM` stable loop with
     frozen run/evaluation evidence;
   - the CE-CNSL OpenSpec is explicitly reactivated after reviewing what CSL-Daily established about shared and
     dataset-specific interfaces.

Neither elapsed time nor free compute automatically satisfies either gate.

This workspace owns CE-CNSL intake and processed delivery. No separate CE-CNSL workspace is created. Shared code
belongs under `src/mmprism/`; dataset-specific filename, label, and source-layout behavior belongs in a thin adapter.

## Dataset Boundary

- Stable ID: `DATASET-CE-CNSL`.
- Preserve `gloss_raw`, versioned `gloss_normalized`, `regional_note`, and `spoken_chinese` separately.
- Keep official train/dev/test assignments for benchmark reproduction.
- Treat the official split as non-participant-disjoint: all 12 signer labels occur in every split.
- Do not call CE-CNSL `CSL-Daily-v2`, silently concatenate datasets, or publish one mixed aggregate score.
- Visual background diversity is not evidence of radar multipath diversity after skeleton-only simulation.

## Gate 1A: Authorized Archive Transfer

After optional download authorization or full activation, download the complete archive into a versioned `incoming/`
or `external/` destination configured through the data root. Record source URL, retrieval time, repository revision,
archive byte count, and SHA-256, then publish a no-clobber immutable archive receipt.

Under download-only authorization, stop here. Do not extract videos, scan codecs, repair labels, implement adapters,
select a pilot, or allocate GPU capacity.

## Gate 1B: Activated Source Intake Audit

After full activation, extract into a versioned destination and record:

- complete extracted inventory and exact train/dev/test CSV checksums;
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
