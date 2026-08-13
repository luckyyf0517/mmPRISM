# CE-CNSL / CE-CSL (2025)

## Citation

Qidan Zhu, Jing Li, Fei Yuan, Jiaojiao Fan, and Quan Gan. "A Chinese Continuous Sign Language Dataset
Based on Complex Environments." arXiv:2409.11960v2, 2025.

- arXiv: <https://arxiv.org/abs/2409.11960>
- Official code and labels: <https://github.com/woshisad159/TFNet>
- Paper name: `CE-CNSL`; the v2 abstract and repository also use `CE-CSL`. They refer to the same dataset.
- Local PDF SHA-256: `e5674d91acc9a73ea082a808ea80bdddd629bed9a85cc0145f88b51115ed1d4f`
- Audited paper version: arXiv v2, updated 2025-09-10.

The local PDF is intentionally untracked under `docs/literature/sources/`. This note separates claims in the
paper from later repository or issue statements.

## Dataset Facts

| Property | CE-CNSL |
|---|---:|
| Videos | 5,988 |
| Duration | 10.52 hours |
| Signers | 12 (A--L) |
| Official split | 4,973 train / 515 dev / 500 test |
| Paper vocabulary | 3,515 Gloss types |
| Sequence length | 39--530 frames |
| Capture | RGB, varying phone, resolution, attire, lighting, and background |
| Environments | More than 70 daily-life settings |
| Labels | Chinese sentence, Gloss sequence, and regional-sign `Note` |

The paper says signers I and J are deaf or hard-of-hearing and describes the other ten as professional sign
language interpreters. It also says professional interpreters reviewed and corrected the videos and labels. The
authors later clarified that the annotations mix standard CSL with a northern regional sign system. Regional
forms are therefore part of the intended dataset rather than automatically erroneous labels.

## Relationship To CSL-Daily

### What the paper establishes

The paper positions CE-CNSL as a response to a limitation of CSL-Daily and similar benchmarks: CSL-Daily was
recorded under controlled laboratory conditions, whereas CE-CNSL deliberately varies devices, clothing,
lighting, and everyday backgrounds. Its Table I gives the following comparison.

| Dataset | Videos | Hours | Signers | Gloss vocabulary | Capture |
|---|---:|---:|---:|---:|---|
| CSL-Daily | 20,654 | 23.27 | 10 | 2,000 | 1920x1080, RGB+D, 30 FPS, lab |
| CE-CNSL | 5,988 | 10.52 | 12 | 3,515 | RGB, varying format/FPS, daily-life scenes |

Thus CE-CNSL is smaller in samples and duration but substantially broader in Gloss vocabulary and visual
capture domain. The two datasets are complementary: CSL-Daily supplies repeated, controlled samples and
CE-CNSL supplies semantic long-tail and source-domain diversity.

The paper trains and evaluates TFNet on the two benchmarks separately. It reports CSL-Daily dev/test WER of
25.1%/23.5% and CE-CNSL dev/test WER of 42.1%/41.9%. These numbers do not constitute a transfer experiment:
the vocabulary, split, and difficulty differ, so the higher CE-CNSL WER must not be attributed solely to complex
backgrounds.

### What later author statements add

The paper does not report a `CSL-Daily -> CE-CNSL` pre-training experiment. In repository issue #1, the author
later recommended pre-training on CSL-Daily and then training on CE-CNSL, stating that CE-CNSL test WER can
reach about 32%. The repository currently advertises a latest weight at 32.46% test WER. This is useful evidence
for the direction of transfer, but it is not a controlled result in the paper: the exact split, vocabulary build,
checkpoint provenance, and training recipe must be independently reproduced before citation as a project result.

- Author statement: <https://github.com/woshisad159/TFNet/issues/1>
- Repository result: <https://github.com/woshisad159/TFNet#inference>

This evidence supports using CSL-Daily as the stable base and CE-CNSL as a subsequent vocabulary/domain
adaptation dataset. It does not support replacing CSL-Daily with CE-CNSL.

## Vocabulary Expansion Audit

A local audit used the repository CSV labels and the CSL-Daily metadata distributed in the TFNet repository.
Counts depend on normalization and are not substitutes for the paper's official statistics.

| Audit item | Result |
|---|---:|
| CSL-Daily Gloss vocabulary | 2,000 |
| CE-CNSL raw Gloss vocabulary | 3,862 |
| CE-CNSL normalized vocabulary | 3,516 |
| Normalized CE-CNSL types also in CSL-Daily | 1,337 |
| Normalized CE-CNSL types new to CSL-Daily | 2,179 |
| Combined normalized union | 4,179 |
| Exact Chinese-sentence overlap | 13 |
| Exact normalized Gloss-sequence overlap | 0 |

The expansion is therefore material: under this normalization, CE-CNSL adds about 2,179 Gloss types and more
than doubles the 2,000-type CSL-Daily vocabulary. It also contributes almost entirely new sentences. This is a
strong reason to use it for semantic expansion, especially for WaveLLM training with full Chinese sentence
targets.

Normalization must remain reversible. The repository preprocessing removes gesture-version suffixes such as
`1`/`2`, parenthesized subject/object information, and bracketed direction information. mmPRISM should instead
retain at least:

- `gloss_raw`: the published sequence without destructive edits;
- `gloss_normalized`: a versioned mapping used for a named experiment;
- `regional_note`: the published regional-sign annotation;
- `spoken_chinese`: the complete Chinese sentence target.

For a fixed CTC Gloss classifier, vocabulary expansion requires an expanded output head or a staged remapping.
For WaveLLM's generative Chinese output, the full sentence labels are directly useful, while Gloss remains an
auxiliary target and audit layer.

## Data And Evaluation Risks

1. The official train, dev, and test splits all contain all 12 signers. They are not participant-disjoint and
   cannot by themselves support a new-user generalization claim.
2. The corpus contains 5,979 unique Chinese sentences among 5,988 videos. Unlike CSL-Daily, it has almost no
   same-sentence repetition across people, so sentence content, signer, device, and background may be confounded.
3. GitHub issue #3 records an acknowledged old-label problem: signer identifiers from H onward did not match the
   video directories. The author stated that sample-number-to-label alignment was unaffected. Signer metadata must
   still be repaired and manually checked before any participant-based split.
4. The official training code builds its Gloss dictionary from train, dev, and test labels together. A new
   implementation must build the vocabulary from train only and declare its OOV policy.
5. The official preprocessing stretches every frame to 256x256 and silently catches frame errors. The mmPRISM
   path should preserve aspect ratio, record decode failures, and use the same confidence/validity contract as
   CSL-Daily `annotation_v2`.
6. The paper reports CSLR WER, not a validated SLT baseline. Natural-language translation performance must be
   established independently.
7. No explicit repository license was found. The paper's ethics statement says the project uses public data under
   an exemption, but it does not clearly document participant consent or redistribution terms. Derived pose or
   simulated-radar redistribution requires written clarification from the authors.
8. Visual background diversity does not automatically become radar environment diversity. If mmPRISM simulates
   only the recovered skeleton, visual backgrounds affect pose-recovery difficulty but do not model radar multipath.

## Recommended Role In mmPRISM

**Conditional go:** admit CE-CNSL as an independent public data family for vocabulary and heterogeneous-domain
adaptation after a pose-recovery pilot. Do not name it CSL-Daily v2, concatenate it invisibly with CSL-Daily, or
use it as a substitute for new real-radar CSL collection.

The execution order should remain:

1. Complete the CSL-Daily `annotation_v2 -> synthetic FMCW -> OmniHand -> WaveLLM` baseline.
2. Receipt the complete CE-CNSL source, freeze checksums, and audit video/CSV/signer correspondence.
3. Run a 120--240-video pose pilot stratified across all 12 signers, devices, resolutions, sequence lengths, and
   difficult backgrounds using the CSL-Daily `annotation_v2` contract.
4. If the pilot passes, process CE-CNSL as a separate manifest and train `CSL-Daily -> CE-CNSL` sequential
   adaptation. Compare CE-CNSL-only and balanced joint/rehearsal training without delaying real-data collection.
5. Report CSL-Daily and CE-CNSL metrics separately. An optional repaired participant-disjoint CE-CNSL split is an
   internal generalization analysis and must not be mixed with official-split WER.

The minimum informative experiment matrix is:

| Training | Evaluation | Question |
|---|---|---|
| CSL-Daily only | CSL-Daily official/control split | Stable baseline and regression anchor |
| CE-CNSL only | CE-CNSL official split | Independent CE-CNSL baseline |
| CSL-Daily then CE-CNSL | Both datasets, separately | Vocabulary/domain adaptation and forgetting |
| Balanced CSL-Daily + CE-CNSL | Both datasets, separately | Joint coverage versus sequential adaptation |

## Project Interpretation

CE-CNSL's vocabulary expansion is beneficial and is the primary reason to adopt it. The correct framing is not
"the distributions differ, therefore do not use it," but "the distributions differ, therefore use it as an
explicit second domain with separate manifests and metrics." It can strengthen evidence about semantic coverage,
public-data scalability, and visual-to-synthetic transfer. It cannot resolve reviewer questions about real radar,
participant-disjoint real-world generalization, off-axis/occlusion sensing, or synthetic-to-real radar fidelity.

This page is a literature note, not project Authority. Dataset admission and execution priority require an explicit
decision after source-license and pose-pilot gates pass.
