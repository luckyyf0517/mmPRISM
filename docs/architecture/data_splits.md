# Canonical Data Split Contract

Status: `split_v1_implemented_partial_real_evidence`
Last Updated: `2026-08-11`
Role: `architecture_contract`

## Purpose

Canonical splits are immutable assignments derived from a pinned data manifest. They never store machine
paths or infer relationships from directory order. A split snapshot is valid only for the exact source
manifest SHA-256, dataset ID, record count, grouping selector, protocol ID and seed recorded in its config.

## Assignment Contract

`mmprism split CONFIG` reads `mmprism.sample.v1` records and selects one explicit grouping value:

- `subject_id` or `group_keys.signer` when identity metadata exists;
- `sequence_id` for sequence-disjoint evidence;
- another declared `group_keys.<name>` for scene/session/acquisition grouping;
- `sample_id` only when independent samples are scientifically justified.

The raw grouping value is not written to the assignment file. A stable full SHA-256 `group_id` is derived
from the grouping namespace, dataset, selector and value. Assignment uses
`sha256_mod_weight_v1`: SHA-256 over protocol ID, integer seed and `group_id`, followed by integer modulo
over ordered positive split weights. No random-number library or floating-point ratio is involved.

Each `mmprism.split_assignment.v1` JSONL record contains exactly:

```json
{"schema_version":"mmprism.split_assignment.v1","sample_id":"...","group_id":"...","split":"train"}
```

## Validation And Finalization

The builder rejects dirty Git state, source manifest hash/count/dataset drift, missing grouping values,
duplicate sample IDs, unknown split names, insufficient per-split group coverage, unsafe snapshot IDs and
insufficient disk space. Before atomic rename it proves:

- every source sample occurs exactly once;
- every `group_id` occurs in exactly one split;
- every configured split has the required number of groups;
- assignment/config/summary checksums are recorded in `SHA256SUMS`.

`SplitIndex` provides dependency-light lookup and split membership without importing PyTorch, Lightning or
Transformers. `mmprism prepare` and each canonical formal train/evaluate service require one explicit
manifest-to-split binding, reject samples missing from the assignment file or assigned to a different split,
and reject overlap between simultaneously supplied manifests. The assignment file is a hashed `split` input
in every resulting `inputs.json`; formal training therefore binds both manifest SHA-256 and assignment
SHA-256 rather than relying on manifest filenames or CLI split labels.

## Partial Versus Final

A split inherits its source scope. A split built from a partial manifest is pipeline evidence only and must
remain marked `partial`; it cannot be promoted by later adding records in place. A complete source requires a
new immutable split snapshot. Subject-independent claims additionally require verified subject/signer metadata;
a sequence-disjoint split cannot substitute for that evidence.
