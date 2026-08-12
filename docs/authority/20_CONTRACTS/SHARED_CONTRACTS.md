# Shared Contracts

Status: current
Owner: mmPRISM coordinator
Authority scope: Minimum contracts that apply across two or more mmPRISM business workspaces.
Last reviewed: 2026-08-12

## Data Identity

- Raw data is immutable; repaired or derived data uses versioned destinations.
- Modality relationships come from validated manifests, never path string replacement.
- A split binds stable sample/group identities and the exact source manifest hash.
- Machine-specific roots, devices, precision, and model locations are runtime configuration.

Detailed schemas remain in [tensor contracts](TENSOR_CONTRACTS.md) and
[data split contracts](DATA_SPLITS.md).

## Formal Runs

Every formal run records resolved configuration, Git state, environment, seed, input hashes, manifest and
split identity, checkpoint or adapter identity, sample-level predictions, and versioned metrics. The local
immutable artifact is the provenance source of truth; an online tracker is a visualization copy.

See [run artifact contracts](RUN_ARTIFACTS.md).

## Frozen Delivery

A cross-workspace delivery identifies:

```text
producer workspace
producer commit
immutable location
manifest or inventory hash
validation status
```

The producer places this identity in an existing manifest summary, receipt, or dated Log. Consumers refer
to that identity instead of duplicating it in a separate handoff document.

## Paper Promotion

A paper-facing metric or claim is promoted only after `paper_revision` can trace it through dataset/split,
resolved run configuration, checkpoint, sample-level output, metric protocol, and manuscript location.
