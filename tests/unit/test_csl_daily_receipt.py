from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from mmprism.cli import main
from mmprism.data import (
    CslDailySourceReceiptError,
    create_csl_daily_source_receipt,
    validate_csl_daily_source_receipt,
)


def _source_tree(root: Path) -> Path:
    source = root / "external" / "csl_daily_original_20260812"
    (source / "CSL-Daily" / "sentence" / "images" / "S000000_P0000_T00").mkdir(
        parents=True
    )
    (source / "CSL-Daily" / "sentence" / "images" / "S000000_P0000_T00" / "000000.jpg").write_bytes(
        b"fixture-frame"
    )
    (source / "CSL-Daily" / "sentence_label").mkdir()
    (source / "CSL-Daily" / "sentence_label" / "csl2020ct_v2.pkl").write_bytes(
        b"fixture-annotation"
    )
    (source / "csl_daily_sentence-crop.zip").write_bytes(b"fixture-archive")
    return source


def _legacy_splits(root: Path) -> Path:
    splits = root / "dataset" / "csl-daily"
    splits.mkdir(parents=True)
    (splits / "all.json").write_text("[\"all\"]\n", encoding="utf-8")
    (splits / "train.json").write_text("[\"train\"]\n", encoding="utf-8")
    (splits / "val.json").write_text("[\"legacy\"]\n", encoding="utf-8")
    (splits / "test.json").write_text("[\"legacy\"]\n", encoding="utf-8")
    return splits


def test_direct_preservation_receipt_is_read_only_and_receipts_legacy_split_identity(
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)
    legacy_splits = _legacy_splits(tmp_path)
    original_paths = sorted(
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    )

    result = create_csl_daily_source_receipt(
        source,
        tmp_path / "interim" / "csl_daily" / "source_receipts",
        source_id="DATASET-CSL-DAILY",
        legacy_split_root=legacy_splits,
        stability_wait_seconds=0,
    )

    receipt_root = result["receipt_root"]
    assert isinstance(receipt_root, str)
    receipt_root = Path(receipt_root)
    assert (
        sorted(path.relative_to(source) for path in source.rglob("*") if path.is_file())
        == original_paths
    )
    receipt = json.loads((receipt_root / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["file_count"] == 3
    assert receipt["source"]["candidates"] == {
        "annotation_pickles": ["CSL-Daily/sentence_label/csl2020ct_v2.pkl"],
        "archive_files": ["csl_daily_sentence-crop.zip"],
        "metadata_files": [],
    }
    assert receipt["legacy_splits"]["val_test_byte_identical"] is True
    assert (
        receipt["legacy_splits"]["use_boundary"]
        == "historical_replay_only_legacy_validation_as_test"
    )
    assert validate_csl_daily_source_receipt(receipt_root)["status"] == "passed"

    with pytest.raises(CslDailySourceReceiptError, match="already exists"):
        create_csl_daily_source_receipt(
            source,
            tmp_path / "interim" / "csl_daily" / "source_receipts",
            source_id="DATASET-CSL-DAILY",
            legacy_split_root=legacy_splits,
            stability_wait_seconds=0,
        )


def test_direct_preservation_receipt_rejects_a_tree_that_changes_during_stability_window(
    tmp_path: Path,
) -> None:
    source = _source_tree(tmp_path)

    def mutate_source(_: float) -> None:
        (source / "CSL-Daily" / "transfer-in-progress.part").write_bytes(b"new bytes")

    with pytest.raises(CslDailySourceReceiptError, match="changed during stability window"):
        create_csl_daily_source_receipt(
            source,
            tmp_path / "receipts",
            source_id="DATASET-CSL-DAILY",
            stability_wait_seconds=1,
            sleeper=mutate_source,
        )
    assert not (tmp_path / "receipts").exists()


def test_direct_preservation_receipt_refuses_to_write_inside_source_tree(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)

    with pytest.raises(CslDailySourceReceiptError, match="outside the read-only source tree"):
        create_csl_daily_source_receipt(
            source,
            source / "interim" / "source_receipts",
            source_id="DATASET-CSL-DAILY",
            stability_wait_seconds=0,
        )


def test_direct_preservation_receipt_validator_rejects_inventory_tampering(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    result = create_csl_daily_source_receipt(
        source,
        tmp_path / "receipts",
        source_id="DATASET-CSL-DAILY",
        stability_wait_seconds=0,
    )
    receipt_root = result["receipt_root"]
    assert isinstance(receipt_root, str)
    receipt_root = Path(receipt_root)
    inventory = receipt_root / "inventory.jsonl"
    inventory.write_text(inventory.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(CslDailySourceReceiptError, match="unsupported schema"):
        validate_csl_daily_source_receipt(receipt_root)


def test_direct_preservation_receipt_cli_round_trip(tmp_path: Path) -> None:
    source = _source_tree(tmp_path)
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = main(
            [
                "csl-daily-source-receipt",
                "--source-root",
                str(source),
                "--receipt-root",
                str(tmp_path / "receipts"),
                "--source-id",
                "DATASET-CSL-DAILY",
                "--stability-wait-seconds",
                "0",
            ]
        )
    assert exit_code == 0
    result = json.loads(output.getvalue())
    receipt_root = result["receipt_root"]
    assert isinstance(receipt_root, str)
    validation_output = io.StringIO()
    with redirect_stdout(validation_output):
        validation_exit = main(["csl-daily-source-receipt-validate", receipt_root])
    assert validation_exit == 0
    assert json.loads(validation_output.getvalue())["status"] == "passed"
