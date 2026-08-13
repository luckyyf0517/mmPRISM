## Why

CSL-Daily camera pose and all downstream synthetic products must already be rebuilt from source. CE-CNSL contributes
5,988 videos, heterogeneous daily-life capture, and approximately 2,179 normalized Gloss types not present in
CSL-Daily according to the current label audit. Its eventual adapter should reuse contracts proven by CSL-Daily,
but implementation is deliberately deferred until that end-to-end path is stable.

Treating it as an invisible CSL-Daily extension would hide important split and dialect differences. Treating it as
a completely separate implementation would duplicate the rebuild. It therefore needs an independent data identity
on a parallel, gated lane that reuses source-independent processing contracts.

## What Changes

- Register `DATASET-CE-CNSL` as a paused P1 follow-on source while CSL-Daily remains P0.
- Require an accepted CSL-Daily end-to-end stable loop and explicit coordinator reactivation before any task starts.
- Add immutable source receipt, video/CSV correspondence, signer-repair, and reversible label contracts.
- Add a 120--240-sequence pose pilot using the CSL-Daily `annotation_v2` output and QC semantics.
- Separate dataset adapters from reusable annotation scheduling, pose payload/QC, simulation, and delivery code.
- Permit full processing only after source/license, identity, and pilot gates pass.
- Define `CSL-Daily -> CE-CNSL` as the first adaptation experiment and require per-dataset reporting.

## Non-Goals

- Replacing or delaying the CSL-Daily P0 reproduction path.
- Downloading, adapting, piloting, or scheduling CE-CNSL before its activation gate.
- Replacing new participant-disjoint real-radar CSL collection.
- Renaming CE-CNSL as CSL-Daily v2 or publishing a mixed aggregate score.
- Reproducing TFNet or accepting its advertised 32.46% checkpoint as an mmPRISM result.
- Publishing source or derived data before license and redistribution permission are established.

## Impact

- The Data Rebuild workspace gains one runbook and an independent dataset/split registry entry.
- `src/mmprism/` gains reusable boundaries only where CSL-Daily and CE-CNSL demonstrably share behavior; each source
  retains a thin layout/label adapter.
- OmniHand and WaveLLM consume the same model-ready contracts with dataset identity and split hashes preserved.
- Paper results distinguish CSL-Daily synthetic controls, CE-CNSL domain expansion, and real-radar evidence.
