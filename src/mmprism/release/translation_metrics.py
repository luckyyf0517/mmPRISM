"""Translation metrics with per-signer corpus scores and segment-weighted pooling."""

from __future__ import annotations

import json

import numpy as np

from mmprism.evaluation.language import (
    TRANSLATION_BLEU_TOKENIZER,
    _official_character_text,
    _official_rouge_l,
)
from mmprism.release.data import SIGNERS
from mmprism.release.paths import PACKAGE_ROOT, resolve
from mmprism.release.settings import write_report


class SimCSE:
    """Pooler-output cosine similarity on text with spaces and newlines removed."""

    def __init__(self, *, device="cpu", batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        path = resolve("assets.simcse")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(self.device)

    def _embed(self, texts):
        chunks = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                pooled = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                ).pooler_output
                chunks.append(pooled.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def per_segment(self, predictions, references):
        def clean(text):
            return (text or "").replace(" ", "").replace("\n", "")

        a = self._embed([clean(text) for text in predictions])
        b = self._embed([clean(text) for text in references])
        num = (a * b).sum(axis=1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        return num / np.clip(den, 1e-12, None)


def score_translation(signers, *, mode, reader, device="cpu"):
    from sacrebleu.metrics import BLEU
    from sentence_transformers import SentenceTransformer

    signers = tuple(dict.fromkeys(signers))
    sbert = SentenceTransformer(str(resolve("assets.sbert")), device=device, local_files_only=True)
    sbert.eval()
    simcse = SimCSE(device=device)
    per_person = {}
    for signer in signers:
        path = resolve("outputs.translation", signer=signer, mode=mode, reader=reader)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        refs = [row["reference"] for row in rows]
        preds = [row["prediction"] for row in rows]
        refs_t = [_official_character_text(text, role="reference") for text in refs]
        preds_t = [_official_character_text(text, role="prediction") for text in preds]
        bleu4 = (
            BLEU(
                tokenize=TRANSLATION_BLEU_TOKENIZER,
                smooth_method="exp",
                effective_order=False,
                max_ngram_order=4,
            )
            .corpus_score(preds_t, [refs_t])
            .score
        )
        rouge = 100.0 * sum(_official_rouge_l(r, p) for r, p in zip(refs_t, preds_t)) / len(preds_t)
        ep = sbert.encode(preds_t, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)
        er = sbert.encode(refs_t, normalize_embeddings=True, batch_size=32, convert_to_numpy=True)
        per_person[signer] = {
            "segments": len(rows),
            "bleu4": float(bleu4),
            "rouge_l": float(rouge),
            "sbert": float(np.mean((ep * er).sum(1))),
            "simcse": 100.0 * float(np.mean(simcse.per_segment(preds, refs))),
        }
        write_report(
            resolve("outputs.translation_signer_report", mode=mode, reader=reader, signer=signer),
            per_person[signer],
        )
    total = sum(row["segments"] for row in per_person.values())
    keys = ("bleu4", "rouge_l", "sbert", "simcse")
    pooled = {
        key: sum(row[key] * row["segments"] for row in per_person.values()) / total for key in keys
    }
    reference = json.loads((PACKAGE_ROOT / "configs/translation_reference.json").read_text())
    if set(signers) == set(SIGNERS):
        pooled["relative_performance"] = (
            100.0 * sum(pooled[key] / reference["metrics"][key] for key in keys) / 4
        )
    result = {
        "mode": mode,
        "reader": reader,
        "segments": total,
        "pooled": pooled,
        "per_person": per_person,
        "reference": reference,
        "aggregation": "segment-weighted mean of per-signer corpus scores",
    }
    if set(signers) == set(SIGNERS):
        write_report(resolve("outputs.translation_report", mode=mode, reader=reader), result)
    return result
