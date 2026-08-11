from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import unicodedata
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from mmprism.data.csl_news import csl_news_source_program

METADATA_PROFILE_SCHEMA_VERSION = "mmprism.csl_news_metadata_profile.v1"
REQUIRED_FIELDS = ("video", "pose", "text")
TOP_CHARACTER_COUNT = 50


class CslNewsMetadataError(ValueError):
    """Raised when a CSL-News metadata profile cannot be built safely."""


def _require_complete_file(path: str | Path, description: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.name.endswith(".part"):
        raise CslNewsMetadataError(
            f"{description} must be complete, not a .part file: {resolved}"
        )
    if not resolved.is_file():
        raise CslNewsMetadataError(f"{description} does not exist: {resolved}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_records(path: Path) -> list[object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload: object = json.load(stream)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CslNewsMetadataError(f"Invalid CSL-News labels JSON: {error}") from error
    if not isinstance(payload, list):
        raise CslNewsMetadataError("CSL-News labels JSON must contain a list")
    return payload


def _load_dataset_card(path: Path) -> tuple[str, Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CslNewsMetadataError(f"Unable to read CSL-News dataset card: {error}") from error

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise CslNewsMetadataError("CSL-News dataset card has no YAML front matter")
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as error:
        raise CslNewsMetadataError(
            "CSL-News dataset card YAML front matter is not closed"
        ) from error
    try:
        raw_front_matter: object = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        raise CslNewsMetadataError(
            f"Invalid CSL-News dataset card YAML front matter: {error}"
        ) from error
    if not isinstance(raw_front_matter, Mapping):
        raise CslNewsMetadataError("CSL-News dataset card front matter must be a mapping")
    return text, raw_front_matter


def _normalize_translation(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return any(
        lower <= codepoint <= upper
        for lower, upper in (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2EBEF),
            (0x30000, 0x323AF),
        )
    )


def _is_lexical(character: str) -> bool:
    return unicodedata.category(character)[0] not in {"P", "Z", "C"}


def _rank_value(sorted_counts: list[tuple[int, int]], rank: int) -> int:
    cumulative = 0
    for value, count in sorted_counts:
        cumulative += count
        if rank < cumulative:
            return value
    raise CslNewsMetadataError("Unable to resolve a length-distribution rank")


def _distribution(counts: Counter[int]) -> dict[str, Any] | None:
    total = counts.total()
    if total == 0:
        return None
    ordered = sorted(counts.items())

    def quantile(probability: float) -> float:
        position = probability * (total - 1)
        lower_rank = math.floor(position)
        upper_rank = math.ceil(position)
        lower = _rank_value(ordered, lower_rank)
        upper = _rank_value(ordered, upper_rank)
        return float(lower + (upper - lower) * (position - lower_rank))

    return {
        "count": total,
        "min": ordered[0][0],
        "max": ordered[-1][0],
        "mean": sum(value * count for value, count in ordered) / total,
        "quantiles": {
            "p01": quantile(0.01),
            "p05": quantile(0.05),
            "p25": quantile(0.25),
            "median": quantile(0.50),
            "p75": quantile(0.75),
            "p95": quantile(0.95),
            "p99": quantile(0.99),
        },
    }


def _top_characters(counts: Counter[str]) -> list[dict[str, Any]]:
    return [
        {
            "character": character,
            "codepoint": f"U+{ord(character):04X}",
            "count": count,
        }
        for character, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:TOP_CHARACTER_COUNT]
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _cross_check_csv(
    csv_path: Path,
    json_records: list[object],
) -> dict[str, Any]:
    expected_by_video: dict[str, Mapping[str, Any]] = {}
    for record in json_records:
        if not isinstance(record, Mapping):
            continue
        video = record.get("video")
        if isinstance(video, str) and video.strip():
            expected_by_video[video] = record

    mismatch_examples: list[dict[str, Any]] = []
    csv_video_counts: Counter[str] = Counter()
    canonical_match_counts: Counter[str] = Counter()
    conflicting_content_row_count = 0
    unknown_video_row_count = 0
    invalid_csv_row_count = 0
    rowwise_mismatch_count = 0
    row_count = 0
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames or []
            for index, row in enumerate(reader):
                row_count += 1
                video = row.get("video")
                pose = row.get("pose")
                text = row.get("text")
                if not all(
                    isinstance(value, str) and value.strip()
                    for value in (video, pose, text)
                ):
                    invalid_csv_row_count += 1
                    if len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            {"index": index, "reason": "CSV row has an invalid field"}
                        )
                    continue
                assert isinstance(video, str)
                csv_video_counts[video] += 1
                expected = expected_by_video.get(video)
                if expected is None:
                    unknown_video_row_count += 1
                    if len(mismatch_examples) < 20:
                        mismatch_examples.append(
                            {
                                "index": index,
                                "reason": "CSV video key is absent from JSON",
                                "video": video,
                            }
                        )
                else:
                    differing_fields = [
                        field
                        for field in REQUIRED_FIELDS
                        if row.get(field) != expected.get(field)
                    ]
                    if differing_fields:
                        conflicting_content_row_count += 1
                        if len(mismatch_examples) < 20:
                            mismatch_examples.append(
                                {
                                    "index": index,
                                    "reason": "CSV content conflicts with canonical JSON",
                                    "video": video,
                                    "fields": differing_fields,
                                }
                            )
                    else:
                        canonical_match_counts[video] += 1

                if index >= len(json_records):
                    rowwise_mismatch_count += 1
                else:
                    json_record = json_records[index]
                    if not isinstance(json_record, Mapping) or any(
                        row.get(field) != json_record.get(field)
                        for field in REQUIRED_FIELDS
                    ):
                        rowwise_mismatch_count += 1
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise CslNewsMetadataError(f"Unable to cross-check CSL-News labels CSV: {error}") from error

    record_count_difference = row_count - len(json_records)
    missing_video_keys = sorted(set(expected_by_video) - set(csv_video_counts))
    unknown_video_keys = sorted(set(csv_video_counts) - set(expected_by_video))
    canonical_record_missing_keys = sorted(
        video for video in expected_by_video if canonical_match_counts[video] == 0
    )
    duplicate_video_keys = sorted(
        video for video, count in csv_video_counts.items() if count > 1
    )
    duplicate_row_count = sum(
        count - 1 for count in csv_video_counts.values() if count > 1
    )
    exact_match = (
        fieldnames == list(REQUIRED_FIELDS)
        and record_count_difference == 0
        and invalid_csv_row_count == 0
        and not canonical_record_missing_keys
        and not unknown_video_keys
        and not duplicate_video_keys
        and conflicting_content_row_count == 0
        and rowwise_mismatch_count == 0
    )
    return {
        "fieldnames": fieldnames,
        "expected_fieldnames": list(REQUIRED_FIELDS),
        "fieldnames_match": fieldnames == list(REQUIRED_FIELDS),
        "row_count": row_count,
        "json_record_count": len(json_records),
        "record_count_difference": record_count_difference,
        "unique_video_count": len(csv_video_counts),
        "key_set_match": not missing_video_keys and not unknown_video_keys,
        "missing_video_key_count": len(missing_video_keys),
        "missing_video_key_examples": missing_video_keys[:20],
        "unknown_video_key_count": len(unknown_video_keys),
        "unknown_video_key_examples": unknown_video_keys[:20],
        "canonical_json_record_present_count": (
            len(expected_by_video) - len(canonical_record_missing_keys)
        ),
        "canonical_json_record_missing_count": len(canonical_record_missing_keys),
        "canonical_json_record_missing_examples": canonical_record_missing_keys[:20],
        "duplicate_video_key_count": len(duplicate_video_keys),
        "duplicate_video_key_examples": duplicate_video_keys[:20],
        "duplicate_row_count": duplicate_row_count,
        "conflicting_content_row_count": conflicting_content_row_count,
        "unknown_video_row_count": unknown_video_row_count,
        "invalid_csv_row_count": invalid_csv_row_count,
        "rowwise_mismatch_count": rowwise_mismatch_count,
        "mismatch_examples": mismatch_examples,
        "exact_match": exact_match,
    }


def build_csl_news_metadata_profile(
    labels_json_path: str | Path,
    labels_csv_path: str | Path,
    dataset_card_path: str | Path,
    *,
    source_id: str,
    source_revision: str,
) -> dict[str, Any]:
    """Validate and characterize the pinned CSL-News label metadata."""

    if not source_id.strip():
        raise CslNewsMetadataError("source_id must be a non-empty string")
    if not source_revision.strip():
        raise CslNewsMetadataError("source_revision must be a non-empty string")
    labels_json = _require_complete_file(labels_json_path, "CSL-News labels JSON")
    labels_csv = _require_complete_file(labels_csv_path, "CSL-News labels CSV")
    dataset_card = _require_complete_file(dataset_card_path, "CSL-News dataset card")
    records = _load_json_records(labels_json)
    card_text, card_front_matter = _load_dataset_card(dataset_card)

    field_set_counts: Counter[tuple[str, ...]] = Counter()
    field_coverage: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    pose_counts: Counter[str] = Counter()
    video_extensions: Counter[str] = Counter()
    pose_extensions: Counter[str] = Counter()
    program_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    all_non_whitespace_characters: Counter[str] = Counter()
    lexical_characters: Counter[str] = Counter()
    han_characters: Counter[str] = Counter()
    normalized_translations: set[str] = set()
    codepoint_lengths: Counter[int] = Counter()
    non_whitespace_lengths: Counter[int] = Counter()
    lexical_lengths: Counter[int] = Counter()
    han_lengths: Counter[int] = Counter()
    invalid_record_examples: list[dict[str, Any]] = []
    invalid_record_count = 0
    empty_translation_count = 0
    normalization_changed_count = 0
    video_pose_stem_mismatch_count = 0
    nested_video_path_count = 0
    nested_pose_path_count = 0
    valid_record_count = 0

    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            invalid_record_count += 1
            if len(invalid_record_examples) < 20:
                invalid_record_examples.append(
                    {"index": index, "reason": "record is not a mapping"}
                )
            continue
        field_set_counts[tuple(sorted(str(key) for key in item))] += 1
        for field in REQUIRED_FIELDS:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                field_coverage[field] += 1
        video = item.get("video")
        pose = item.get("pose")
        text = item.get("text")
        invalid_fields = [
            field
            for field, value in (("video", video), ("pose", pose), ("text", text))
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_fields:
            invalid_record_count += 1
            if "text" in invalid_fields:
                empty_translation_count += 1
            if len(invalid_record_examples) < 20:
                invalid_record_examples.append(
                    {
                        "index": index,
                        "reason": "required field is not a non-empty string",
                        "fields": invalid_fields,
                    }
                )
            continue

        assert isinstance(video, str)
        assert isinstance(pose, str)
        assert isinstance(text, str)
        normalized_video = video.strip().replace("\\", "/")
        normalized_pose = pose.strip().replace("\\", "/")
        video_path = PurePosixPath(normalized_video)
        pose_path = PurePosixPath(normalized_pose)
        video_name = video_path.name
        pose_name = pose_path.name
        if normalized_video != video_name:
            nested_video_path_count += 1
        if normalized_pose != pose_name:
            nested_pose_path_count += 1
        video_counts[video_name] += 1
        pose_counts[pose_name] += 1
        video_extensions[video_path.suffix.lower()] += 1
        pose_extensions[pose_path.suffix.lower()] += 1
        program_counts[csl_news_source_program(video_name)] += 1
        if video_path.stem != pose_path.stem:
            video_pose_stem_mismatch_count += 1

        normalized_text = _normalize_translation(text)
        if normalized_text != text:
            normalization_changed_count += 1
        if not normalized_text:
            empty_translation_count += 1
            invalid_record_count += 1
            if len(invalid_record_examples) < 20:
                invalid_record_examples.append(
                    {"index": index, "reason": "translation is empty after normalization"}
                )
            continue
        normalized_translations.add(normalized_text)
        codepoint_lengths[len(normalized_text)] += 1
        non_whitespace_count = 0
        lexical_count = 0
        han_count = 0
        for character in normalized_text:
            category_counts[unicodedata.category(character)] += 1
            if not character.isspace():
                non_whitespace_count += 1
                all_non_whitespace_characters[character] += 1
            if _is_lexical(character):
                lexical_count += 1
                lexical_characters[character] += 1
            if _is_han(character):
                han_count += 1
                han_characters[character] += 1
        non_whitespace_lengths[non_whitespace_count] += 1
        lexical_lengths[lexical_count] += 1
        han_lengths[han_count] += 1
        valid_record_count += 1

    csv_cross_check = _cross_check_csv(labels_csv, records)
    duplicate_video_keys = sum(count > 1 for count in video_counts.values())
    duplicate_pose_keys = sum(count > 1 for count in pose_counts.values())
    duplicate_video_records = sum(count - 1 for count in video_counts.values() if count > 1)
    duplicate_pose_records = sum(count - 1 for count in pose_counts.values() if count > 1)

    languages = _string_list(card_front_matter.get("language"))
    task_categories = _string_list(card_front_matter.get("task_categories"))
    license_value = card_front_matter.get("license")
    license_name = license_value if isinstance(license_value, str) else None
    declared_sign_language = (
        "Chinese Sign Language" if "Chinese Sign Language" in card_text else None
    )
    field_names = set().union(
        *(set(field_set) for field_set in field_set_counts)
    ) if field_set_counts else set()

    failures: list[str] = []
    warnings: list[str] = []
    if invalid_record_count:
        failures.append(f"{invalid_record_count} JSON records are invalid")
    if duplicate_video_keys:
        failures.append(f"{duplicate_video_keys} video keys are duplicated")
    if duplicate_pose_keys:
        failures.append(f"{duplicate_pose_keys} pose keys are duplicated")
    if video_pose_stem_mismatch_count:
        failures.append(
            f"{video_pose_stem_mismatch_count} video/pose filename stems differ"
        )
    if dict(video_extensions) != {".mp4": valid_record_count}:
        failures.append("video extensions are not uniformly .mp4")
    if dict(pose_extensions) != {".pkl": valid_record_count}:
        failures.append("pose extensions are not uniformly .pkl")
    if not csv_cross_check["fieldnames_match"]:
        failures.append("CSV fieldnames differ from the required JSON fields")
    if csv_cross_check["invalid_csv_row_count"]:
        failures.append(
            f"CSV contains {csv_cross_check['invalid_csv_row_count']} invalid rows"
        )
    if csv_cross_check["missing_video_key_count"]:
        failures.append(
            f"CSV is missing {csv_cross_check['missing_video_key_count']} JSON video keys"
        )
    if csv_cross_check["unknown_video_key_count"]:
        failures.append(
            f"CSV contains {csv_cross_check['unknown_video_key_count']} unknown video keys"
        )
    if csv_cross_check["canonical_json_record_missing_count"]:
        failures.append(
            "CSV does not contain the canonical JSON content for "
            f"{csv_cross_check['canonical_json_record_missing_count']} video keys"
        )
    if csv_cross_check["duplicate_video_key_count"]:
        warnings.append(
            f"CSV contains {csv_cross_check['duplicate_video_key_count']} duplicate video "
            f"keys and {csv_cross_check['duplicate_row_count']} extra rows"
        )
    if csv_cross_check["conflicting_content_row_count"]:
        warnings.append(
            f"CSV contains {csv_cross_check['conflicting_content_row_count']} rows whose "
            "content conflicts with the canonical JSON record"
        )
    if csv_cross_check["rowwise_mismatch_count"]:
        warnings.append(
            f"CSV row order/content differs from JSON at "
            f"{csv_cross_check['rowwise_mismatch_count']} positions"
        )
    if "zh" not in languages:
        failures.append("dataset card does not declare language zh")
    if license_name != "cc-by-nc-4.0":
        failures.append("dataset card license is not cc-by-nc-4.0")
    if "video-text-to-text" not in task_categories:
        failures.append("dataset card does not declare video-text-to-text")
    if declared_sign_language is None:
        failures.append("dataset card does not declare Chinese Sign Language")

    return {
        "schema_version": METADATA_PROFILE_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "failed"
            if failures
            else "passed_with_warnings"
            if warnings
            else "passed"
        ),
        "failures": failures,
        "warnings": warnings,
        "source": {
            "source_id": source_id.strip(),
            "source_revision": source_revision.strip(),
            "declared_language_codes": languages,
            "declared_sign_language": declared_sign_language,
            "declared_task_categories": task_categories,
            "declared_license": license_name,
        },
        "files": {
            "labels_json": {
                "path": str(labels_json),
                "size_bytes": labels_json.stat().st_size,
                "sha256": _sha256(labels_json),
            },
            "labels_csv": {
                "path": str(labels_csv),
                "size_bytes": labels_csv.stat().st_size,
                "sha256": _sha256(labels_csv),
            },
            "dataset_card": {
                "path": str(dataset_card),
                "size_bytes": dataset_card.stat().st_size,
                "sha256": _sha256(dataset_card),
            },
        },
        "canonical_annotation_source": {
            "format": "json",
            "path": str(labels_json),
            "selection_rule": (
                "The JSON video key must be unique and is authoritative; CSV is retained "
                "as a cross-check and is never used to silently override JSON."
            ),
        },
        "record_schema": {
            "required_fields": list(REQUIRED_FIELDS),
            "observed_field_sets": [
                {"fields": list(fields), "count": count}
                for fields, count in sorted(field_set_counts.items())
            ],
            "field_coverage_counts": {
                field: field_coverage[field] for field in REQUIRED_FIELDS
            },
            "explicit_field_availability": {
                "sentence_id": "sentence_id" in field_names,
                "sign_or_gloss_label": bool(
                    {"sign", "sign_id", "gloss", "glosses"} & field_names
                ),
                "non_manual_features": bool(
                    {"non_manual", "non_manual_features"} & field_names
                ),
                "subject_or_signer_id": bool(
                    {"subject", "subject_id", "signer", "signer_id"} & field_names
                ),
                "scene_or_environment": bool(
                    {"scene", "scene_id", "environment"} & field_names
                ),
                "orientation": "orientation" in field_names,
                "occlusion": "occlusion" in field_names,
                "split": "split" in field_names,
            },
        },
        "integrity": {
            "json_record_count": len(records),
            "valid_record_count": valid_record_count,
            "invalid_record_count": invalid_record_count,
            "invalid_record_examples": invalid_record_examples,
            "empty_translation_count": empty_translation_count,
            "unique_video_count": len(video_counts),
            "duplicate_video_key_count": duplicate_video_keys,
            "duplicate_video_record_count": duplicate_video_records,
            "unique_pose_count": len(pose_counts),
            "duplicate_pose_key_count": duplicate_pose_keys,
            "duplicate_pose_record_count": duplicate_pose_records,
            "video_pose_stem_mismatch_count": video_pose_stem_mismatch_count,
            "nested_video_path_count": nested_video_path_count,
            "nested_pose_path_count": nested_pose_path_count,
            "video_extension_counts": dict(sorted(video_extensions.items())),
            "pose_extension_counts": dict(sorted(pose_extensions.items())),
            "normalization_changed_translation_count": normalization_changed_count,
            "csv_cross_check": csv_cross_check,
        },
        "dataset_units": {
            "annotation_unit": "video_segment_with_natural_language_translation",
            "translation_segment_count": valid_record_count,
            "unique_normalized_translation_count": len(normalized_translations),
            "repeated_translation_record_count": (
                valid_record_count - len(normalized_translations)
            ),
            "explicit_sentence_count": None,
            "sign_vocabulary_size": None,
            "source_program_counts": dict(sorted(program_counts.items())),
        },
        "translation_statistics": {
            "normalization": "Unicode NFC; trim and collapse Unicode whitespace",
            "length_units": {
                "unicode_codepoints": _distribution(codepoint_lengths),
                "non_whitespace_codepoints": _distribution(non_whitespace_lengths),
                "lexical_codepoints_excluding_unicode_P_Z_C": _distribution(
                    lexical_lengths
                ),
                "han_codepoints": _distribution(han_lengths),
            },
            "vocabularies": {
                "non_whitespace_codepoint_size": len(all_non_whitespace_characters),
                "lexical_codepoint_size": len(lexical_characters),
                "han_character_size": len(han_characters),
                "top_lexical_characters": _top_characters(lexical_characters),
            },
            "unicode_category_counts": dict(sorted(category_counts.items())),
        },
        "limitations": [
            "Each record is a video segment paired with a natural-language translation; "
            "the metadata does not provide an explicit sentence identifier.",
            "Translation character vocabulary is not sign-language gloss or sign "
            "vocabulary and must not be reported as such.",
            "The metadata does not annotate non-manual grammatical features.",
            "The metadata does not identify subjects or signers, scenes, orientations, "
            "occlusions, or official splits.",
            "Program categories are inferred only from filename substrings and unknown "
            "is retained rather than guessed.",
        ],
    }


def write_csl_news_metadata_profile(
    report: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write a CSL-News metadata profile atomically."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
