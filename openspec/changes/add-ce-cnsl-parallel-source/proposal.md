## Why

CSL-Daily camera pose and all downstream synthetic products must already be rebuilt from source. The intake,
manifest, pose-validity, simulation, and delivery capabilities being implemented should support a second CSL source
without waiting for the complete CSL-Daily training cycle. CE-CNSL contributes 5,988 videos, heterogeneous daily-life
capture, and approximately 2,179 normalized Gloss types not present in CSL-Daily according to the current label audit.

Treating it as an invisible CSL-Daily extension would hide important split and dialect differences. Treating it as
a completely separate implementation would duplicate the rebuild. It therefore needs an independent data identity
on a parallel, gated lane that reuses source-independent processing contracts.

## What Changes

- Register `DATASET-CE-CNSL` as a P1 parallel source while CSL-Daily remains P0.
- Add immutable source receipt, video/CSV correspondence, signer-repair, and reversible label contracts.
- Add a 120--240-sequence pose pilot using the CSL-Daily `annotation_v2` output and QC semantics.
- Separate dataset adapters from reusable annotation scheduling, pose payload/QC, simulation, and delivery code.
- Permit full processing only after source/license, identity, and pilot gates pass.
- Define `CSL-Daily -> CE-CNSL` as the first adaptation experiment and require per-dataset reporting.

## Non-Goals

- Replacing or delaying the CSL-Daily P0 reproduction path.
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
