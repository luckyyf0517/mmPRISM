# Uni-Sign (ICLR 2025)

## Citation

Zecheng Li, Wengang Zhou, Weichao Zhao, Kepeng Wu, Hezhen Hu, and Houqiang Li. “Uni-Sign: Toward
Unified Sign Language Understanding at Scale.” ICLR 2025. arXiv:2501.15187.

- arXiv: <https://arxiv.org/abs/2501.15187>
- Official code: <https://github.com/ZechengLi19/Uni-Sign>
- Official CSL-News data: <https://huggingface.co/datasets/ZechengLi19/CSL-News>
- Local PDF SHA-256: `ac6aba55d4613a89e7f723d6de99a6078c97acc5ccd29afb24ce0d1d14a1bca7`
- Local arXiv source archive SHA-256: `549d63229e04b4056ff2e15c7f3efe109b4da8ba60b7b22386584ce53e431172`

The local originals are intentionally untracked and are identified here by checksum rather than a repository link.

## Relevant Facts

- CSL-News contains about 1,985 hours of CSL video-text pairs, a vocabulary of about 5K, and TV/news content.
- CSL-Daily contains about 23 hours and a vocabulary of about 2K in the paper's dataset comparison.
- Uni-Sign uses CSL-News for CSL pre-training in Stages 1 and 2, then performs task-specific downstream
  fine-tuning in Stage 3.
- The paper explicitly describes the scale ablation as random sampling from CSL-News. The table below is the
  pose-only setting evaluated on CSL-Daily.

| CSL-News pre-training share | CSLR Test WER down | SLT Test BLEU-4 up | SLT Test ROUGE up |
|---:|---:|---:|---:|
| 0% | 73.6 | 3.51 | 20.56 |
| 25% | 31.0 | 21.13 | 49.90 |
| 50% | 30.1 | 22.58 | 51.62 |
| 75% | 28.5 | 24.95 | 54.87 |
| 100% | 27.4 | 25.61 | 54.92 |

## Interpretation For mmPRISM

The evidence supports the general expectation that more CSL-News pre-training data can raise the absolute
performance ceiling. It does not establish that mmPRISM's historical checkpoint, trained on the first roughly
100 of 436 archives, is equivalent to Uni-Sign's randomly sampled 25% condition. Archive ordering may introduce
content, broadcaster, signer, or time bias, and the training recipes are not assumed identical.

For the current revision, full CSL-News retraining is therefore a future ceiling experiment rather than a reviewer
requirement. The controlled sim2real question is better isolated by retaining the same original-submission semantic
initialization across architecture, adaptation, and real stress comparisons. The full 436-archive download,
annotation, and manifest would still be useful for source provenance, dataset statistics, simulation input, and a
future full-data ceiling experiment, but they need not be completed or resumed for the revision.

This interpretation was originally promoted as `DEC-044` and its current historical-asset boundary is `DEC-046` in the
[project decision log](../../authority/60_DECISIONS/DECISION_LOG.md).
