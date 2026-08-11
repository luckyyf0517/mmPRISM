#!/usr/bin/env python3
"""Build a deterministic structural and editorial audit of a LaTeX manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

AUDIT_SCHEMA = "mmprism.manuscript_audit.v2"
GRAPHIC_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".eps", ".svg")
SECTION_COMMANDS = ("part", "chapter", "section", "subsection", "subsubsection")
REFERENCE_COMMANDS = ("ref", "eqref", "autoref", "cref", "Cref", "pageref")
CITATION_COMMANDS = (
    "cite",
    "citep",
    "citet",
    "citealp",
    "citeauthor",
    "citeyear",
    "parencite",
    "textcite",
    "autocite",
    "footcite",
)

# The first group mirrors the editor's explicit examples. The second group captures
# evidence-sensitive language already used in the manuscript and requires human review.
SOBER_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("ED-SOBER-NEW", "editor_prohibited", "new", r"\bnew\b"),
    ("ED-SOBER-NOVEL", "editor_prohibited", "novel", r"\bnovel\b"),
    ("ED-SOBER-FIRST", "editor_prohibited", "first", r"\bfirst\b"),
    ("ED-SOBER-UNIQUE", "editor_prohibited", "unique", r"\bunique\b"),
    (
        "ED-SOBER-UNPRECEDENTED",
        "editor_prohibited",
        "unprecedented",
        r"\bunprecedented\b",
    ),
    (
        "ED-SOBER-FIRST-KIND",
        "editor_prohibited",
        "first-of-its-kind",
        r"\bfirst[- ]of[- ](?:its|the)[- ]kind\b",
    ),
    (
        "ED-SOBER-NEW-PARADIGM",
        "editor_prohibited",
        "a new paradigm",
        r"\ba new paradigm\b",
    ),
    ("ED-SOBER-EXTREMELY", "editor_prohibited", "extremely", r"\bextremely\b"),
    ("ED-SOBER-OUTSTANDING", "editor_prohibited", "outstanding", r"\boutstanding\b"),
    ("ED-SOBER-EXCELLENT", "editor_prohibited", "excellent", r"\bexcellent\b"),
    ("ED-SOBER-REMARKABLE", "editor_prohibited", "remarkable", r"\bremarkable\b"),
    ("ED-SOBER-ULTRA", "editor_prohibited", "ultra", r"\bultra\b"),
    ("ED-SOBER-SUPERIOR", "editor_prohibited", "superior", r"\bsuperior\b"),
    ("ED-SOBER-FASCINATING", "editor_prohibited", "fascinating", r"\bfascinating\b"),
    (
        "ED-SOBER-PAVE-WAY",
        "editor_prohibited",
        "pave the way",
        r"\bpav(?:e|es|ed|ing) the way\b",
    ),
    (
        "ED-SOBER-OPEN-AVENUES",
        "editor_prohibited",
        "open new avenues",
        r"\bopen(?:s|ed|ing)? new avenues\b",
    ),
    (
        "ED-SOBER-PAVE-PATH",
        "editor_prohibited",
        "pave a new path/route",
        r"\bpav(?:e|es|ed|ing) (?:a )?(?:new )?(?:path|route)\b",
    ),
    (
        "CLAIM-STATE-OF-ART",
        "evidence_sensitive",
        "state-of-the-art",
        r"\bstate[- ]of[- ]the[- ]art\b",
    ),
    (
        "CLAIM-HIGH-FIDELITY",
        "evidence_sensitive",
        "high-fidelity",
        r"\bhigh[- ]fidelity\b",
    ),
    (
        "CLAIM-OPTICAL-LEVEL",
        "evidence_sensitive",
        "optical-level",
        r"\boptical[- ]level\b",
    ),
    (
        "CLAIM-GENERALIZABLE-PARADIGM",
        "evidence_sensitive",
        "generalizable paradigm",
        r"\bgeneralizable paradigm\b",
    ),
    (
        "CLAIM-SIGNIFICANTLY-OUTPERFORM",
        "evidence_sensitive",
        "significantly outperform",
        r"\bsignificantly outperform(?:s|ed|ing)?\b",
    ),
)

PLACEHOLDER_PATTERNS: tuple[tuple[str, str], ...] = (
    ("PLACEHOLDER-TODO", r"\b(?:TODO|TBD|PLACEHOLDER)\b"),
    (
        "PLACEHOLDER-EXAMPLE-REAL-DATA-CN",
        r"(?:示例[^\r\n%]{0,40})?替换为真实数据",
    ),
)


@dataclass(frozen=True)
class Command:
    name: str
    argument: str
    start: int
    end: int
    starred: bool


@dataclass(frozen=True)
class TexSource:
    path: Path
    relative_path: str
    raw_text: str
    active_text: str
    line_starts: tuple[int, ...]

    def line_number(self, offset: int) -> int:
        return bisect_right(self.line_starts, offset)

    def context(self, offset: int) -> str:
        line = self.active_text.splitlines()[self.line_number(offset) - 1]
        return collapse_whitespace(line)

    def raw_context(self, offset: int) -> str:
        line = self.raw_text.splitlines()[self.line_number(offset) - 1]
        return collapse_whitespace(line)


def collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_latex_comments(text: str) -> str:
    """Replace unescaped comments with spaces while preserving offsets and newlines."""

    output: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at: int | None = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            output.append(line)
            continue
        suffix = line[comment_at:]
        newline = "\n" if suffix.endswith("\n") else ""
        if suffix.endswith("\r\n"):
            newline = "\r\n"
        output.append(line[:comment_at] + " " * (len(suffix) - len(newline)) + newline)
    return "".join(output)


def _extract_delimited(text: str, start: int, opening: str, closing: str) -> tuple[str, int]:
    if start >= len(text) or text[start] != opening:
        raise ValueError(f"expected {opening!r} at offset {start}")
    depth = 0
    cursor = start
    while cursor < len(text):
        character = text[cursor]
        if character == opening and not _is_escaped(text, cursor):
            depth += 1
        elif character == closing and not _is_escaped(text, cursor):
            depth -= 1
            if depth == 0:
                return text[start + 1 : cursor], cursor + 1
        cursor += 1
    raise ValueError(f"unclosed {opening!r} at offset {start}")


def _is_escaped(text: str, offset: int) -> bool:
    slash_count = 0
    cursor = offset - 1
    while cursor >= 0 and text[cursor] == "\\":
        slash_count += 1
        cursor -= 1
    return slash_count % 2 == 1


def iter_commands(text: str, names: Iterable[str]) -> Iterator[Command]:
    wanted = set(names)
    command_re = re.compile(r"\\(?P<name>[A-Za-z@]+)(?P<star>\*)?")
    for match in command_re.finditer(text):
        name = match.group("name")
        if name not in wanted:
            continue
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        try:
            while cursor < len(text) and text[cursor] == "[":
                _, cursor = _extract_delimited(text, cursor, "[", "]")
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                continue
            argument, end = _extract_delimited(text, cursor, "{", "}")
        except ValueError:
            continue
        yield Command(
            name=name,
            argument=argument,
            start=match.start(),
            end=end,
            starred=bool(match.group("star")),
        )


def _source_from_text(relative_path: str, raw: str, *, path: Path | None = None) -> TexSource:
    active = strip_latex_comments(raw)
    line_starts = [0]
    line_starts.extend(match.end() for match in re.finditer("\n", active))
    return TexSource(
        path=path if path is not None else Path(relative_path),
        relative_path=relative_path,
        raw_text=raw,
        active_text=active,
        line_starts=tuple(line_starts),
    )


def _source(path: Path, root: Path) -> TexSource:
    return _source_from_text(
        path.relative_to(root).as_posix(),
        path.read_text(encoding="utf-8"),
        path=path,
    )


def _within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_local_target(
    root: Path,
    source_path: Path,
    target: str,
    extensions: Sequence[str],
) -> Path | None:
    clean_target = collapse_whitespace(target).strip()
    if not clean_target or "\\" in clean_target:
        return None
    target_path = Path(clean_target)
    if target_path.is_absolute():
        return None
    bases = (root, source_path.parent)
    suffixes = ("",) if target_path.suffix else tuple(extensions)
    for base in bases:
        for suffix in suffixes:
            candidate = (base / f"{clean_target}{suffix}").resolve()
            if _within_root(candidate, root) and candidate.is_file():
                return candidate
    return None


def _location(source: TexSource, command: Command) -> dict[str, Any]:
    return {
        "file": source.relative_path,
        "line": source.line_number(command.start),
        "context": source.context(command.start),
    }


def _plain_latex(value: str) -> str:
    result = collapse_whitespace(value).replace("~", " ")
    formatting = re.compile(r"\\(?:textbf|textit|emph|underline|textrm|textsf)\s*\{([^{}]*)\}")
    previous = ""
    while result != previous:
        previous = result
        result = formatting.sub(r"\1", result)
    result = re.sub(r"\\[A-Za-z@]+\*?", " ", result)
    return collapse_whitespace(result.replace("{", " ").replace("}", " "))


def _scan_document_graph(root: Path, entry: Path) -> dict[str, Any]:
    sources: dict[str, TexSource] = {}
    include_edges: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, Any]] = []
    include_cycles: list[list[str]] = []
    section_order: list[dict[str, Any]] = []

    def visit(path: Path, stack: tuple[str, ...]) -> None:
        relative = path.relative_to(root).as_posix()
        if relative in stack:
            include_cycles.append([*stack, relative])
            return
        source = sources.get(relative)
        if source is None:
            source = _source(path, root)
            sources[relative] = source

        events = list(iter_commands(source.active_text, (*SECTION_COMMANDS, "input", "include")))
        for command in sorted(events, key=lambda item: item.start):
            if command.name in SECTION_COMMANDS:
                section_order.append(
                    {
                        "level": command.name,
                        "starred": command.starred,
                        "title": _plain_latex(command.argument),
                        **_location(source, command),
                    }
                )
                continue
            resolved = _resolve_local_target(root, source.path, command.argument, (".tex",))
            edge = {
                "source": relative,
                "line": source.line_number(command.start),
                "target": collapse_whitespace(command.argument),
                "resolved": resolved.relative_to(root).as_posix() if resolved else None,
            }
            include_edges.append(edge)
            if resolved is None:
                missing_inputs.append(edge)
            else:
                visit(resolved, (*stack, relative))

    visit(entry, ())
    return {
        "sources": sources,
        "include_edges": include_edges,
        "missing_inputs": missing_inputs,
        "include_cycles": include_cycles,
        "section_order": section_order,
    }


def _find_environments(source: TexSource, environment_type: str) -> list[dict[str, Any]]:
    begin_re = re.compile(rf"\\begin\s*\{{({environment_type}\*?)\}}")
    sections = list(iter_commands(source.active_text, SECTION_COMMANDS))
    results: list[dict[str, Any]] = []
    for index, begin in enumerate(begin_re.finditer(source.active_text), start=1):
        environment = begin.group(1)
        end_re = re.compile(rf"\\end\s*\{{{re.escape(environment)}\}}")
        end = end_re.search(source.active_text, begin.end())
        content_end = end.end() if end else len(source.active_text)
        content = source.active_text[begin.start() : content_end]
        label_commands = list(iter_commands(content, ("label",)))
        caption_commands = list(iter_commands(content, ("caption",)))
        graphic_commands = list(iter_commands(content, ("includegraphics",)))
        labels = [collapse_whitespace(item.argument) for item in label_commands]
        captions = [collapse_whitespace(item.argument) for item in caption_commands]
        graphics = [collapse_whitespace(item.argument) for item in graphic_commands]
        display_items: list[dict[str, Any]] = []
        if caption_commands:
            for item_index, caption in enumerate(caption_commands, start=1):
                lower_bound = caption_commands[item_index - 2].end if item_index > 1 else 0
                upper_bound = (
                    caption_commands[item_index].start
                    if item_index < len(caption_commands)
                    else len(content)
                )
                item_graphics = [
                    collapse_whitespace(item.argument)
                    for item in graphic_commands
                    if lower_bound <= item.start < caption.start
                ]
                item_labels = [
                    collapse_whitespace(item.argument)
                    for item in label_commands
                    if caption.end <= item.start < upper_bound
                ]
                display_items.append(
                    {
                        "item_index_in_environment": item_index,
                        "environment": environment,
                        "environment_index_in_file": index,
                        "file": source.relative_path,
                        "line": source.line_number(begin.start() + caption.start),
                        "labels": item_labels,
                        "caption": collapse_whitespace(caption.argument),
                        "graphics": item_graphics,
                    }
                )
        else:
            display_items.append(
                {
                    "item_index_in_environment": 1,
                    "environment": environment,
                    "environment_index_in_file": index,
                    "file": source.relative_path,
                    "line": source.line_number(begin.start()),
                    "labels": labels,
                    "caption": None,
                    "graphics": graphics,
                }
            )
        preceding_sections = [item for item in sections if item.start < begin.start()]
        section = _plain_latex(preceding_sections[-1].argument) if preceding_sections else None
        for display_item in display_items:
            display_item["section"] = section
        results.append(
            {
                "index_in_file": index,
                "environment": environment,
                "file": source.relative_path,
                "line": source.line_number(begin.start()),
                "section": section,
                "labels": labels,
                "captions": captions,
                "graphics": graphics,
                "display_items": display_items,
                "closed": end is not None,
            }
        )
    return results


def _assign_display_ids(
    displays: list[dict[str, Any]], prefix: str
) -> list[dict[str, Any]]:
    for index, display in enumerate(displays, start=1):
        display["display_id"] = f"{prefix}-{index:02d}"
    return displays


def _flatten_display_items(environments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for environment in environments for item in environment["display_items"]]


def _document_body(source: TexSource) -> str:
    begin = re.search(r"\\begin\s*\{document\}", source.active_text)
    if begin is None:
        return source.active_text
    end = re.search(r"\\end\s*\{document\}", source.active_text[begin.end() :])
    if end is None:
        return source.active_text[begin.end() :]
    return source.active_text[begin.end() : begin.end() + end.start()]


def _sober_hits(source: TexSource) -> list[dict[str, Any]]:
    body = _document_body(source)
    body_start = source.active_text.find(body)
    hits: list[dict[str, Any]] = []
    for rule_id, category, term, pattern in SOBER_PATTERNS:
        for match in re.finditer(pattern, body, flags=re.IGNORECASE):
            offset = body_start + match.start()
            hits.append(
                {
                    "rule_id": rule_id,
                    "category": category,
                    "term": term,
                    "matched_text": match.group(0),
                    "file": source.relative_path,
                    "line": source.line_number(offset),
                    "context": source.context(offset),
                }
            )
    return sorted(hits, key=lambda item: (item["file"], item["line"], item["rule_id"]))


def _placeholder_hits(source: TexSource) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rule_id, pattern in PLACEHOLDER_PATTERNS:
        for match in re.finditer(pattern, source.raw_text, flags=re.IGNORECASE):
            matched_text = match.group(0)
            hits.append(
                {
                    "rule_id": rule_id,
                    "matched_text": matched_text,
                    "active": source.active_text[match.start() : match.end()] == matched_text,
                    "file": source.relative_path,
                    "line": source.line_number(match.start()),
                    "context": source.raw_context(match.start()),
                }
            )
    return sorted(hits, key=lambda item: (item["file"], item["line"], item["rule_id"]))


def _bibliography_keys(text: str) -> list[str]:
    return re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", strip_latex_comments(text))


def _fingerprint(records: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: str(item["path"])):
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit_manuscript(root: Path, entry_name: str) -> dict[str, Any]:
    root = root.resolve()
    entry = (root / entry_name).resolve()
    if not _within_root(entry, root) or not entry.is_file():
        raise FileNotFoundError(f"manuscript entry does not exist under root: {entry_name}")

    graph = _scan_document_graph(root, entry)
    sources: dict[str, TexSource] = graph.pop("sources")
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    graphics: list[dict[str, Any]] = []
    bibliography_commands: list[dict[str, Any]] = []
    sober_hits: list[dict[str, Any]] = []
    placeholder_hits: list[dict[str, Any]] = []
    citation_command_count = 0
    reference_command_count = 0

    for source in sources.values():
        source_figures = _find_environments(source, "figure")
        for figure in source_figures:
            figure["resolved_graphics"] = [
                (
                    resolved.relative_to(root).as_posix()
                    if (
                        resolved := _resolve_local_target(
                            root, source.path, target, GRAPHIC_EXTENSIONS
                        )
                    )
                    else None
                )
                for target in figure["graphics"]
            ]
            for display_item in figure["display_items"]:
                display_item["resolved_graphics"] = [
                    (
                        resolved.relative_to(root).as_posix()
                        if (
                            resolved := _resolve_local_target(
                                root, source.path, target, GRAPHIC_EXTENSIONS
                            )
                        )
                        else None
                    )
                    for target in display_item["graphics"]
                ]
        figures.extend(source_figures)
        tables.extend(_find_environments(source, "table"))
        sober_hits.extend(_sober_hits(source))
        placeholder_hits.extend(_placeholder_hits(source))
        for command in iter_commands(source.active_text, ("label",)):
            labels.append(
                {"key": collapse_whitespace(command.argument), **_location(source, command)}
            )
        reference_commands = list(iter_commands(source.active_text, REFERENCE_COMMANDS))
        reference_command_count += len(reference_commands)
        for command in reference_commands:
            for key in command.argument.split(","):
                references.append(
                    {
                        "key": key.strip(),
                        "command": command.name,
                        **_location(source, command),
                    }
                )
        citation_commands = list(iter_commands(source.active_text, CITATION_COMMANDS))
        citation_command_count += len(citation_commands)
        for command in citation_commands:
            for key in command.argument.split(","):
                citations.append(
                    {
                        "key": key.strip(),
                        "command": command.name,
                        **_location(source, command),
                    }
                )
        for command in iter_commands(source.active_text, ("includegraphics",)):
            resolved = _resolve_local_target(
                root, source.path, command.argument, GRAPHIC_EXTENSIONS
            )
            graphics.append(
                {
                    "target": collapse_whitespace(command.argument),
                    "resolved": resolved.relative_to(root).as_posix() if resolved else None,
                    **_location(source, command),
                }
            )
        for command in iter_commands(source.active_text, ("bibliography", "addbibresource")):
            for target in command.argument.split(","):
                resolved = _resolve_local_target(root, source.path, target.strip(), (".bib",))
                bibliography_commands.append(
                    {
                        "target": target.strip(),
                        "resolved": resolved.relative_to(root).as_posix() if resolved else None,
                        **_location(source, command),
                    }
                )

    label_counts = Counter(item["key"] for item in labels)
    label_keys = set(label_counts)
    duplicate_labels = sorted(key for key, count in label_counts.items() if count > 1)
    referenced_keys = {item["key"] for item in references}
    missing_references = sorted(referenced_keys - label_keys)
    unreferenced_labels = sorted(label_keys - referenced_keys)

    bibliography_files: list[dict[str, Any]] = []
    all_bibliography_keys: list[str] = []
    for relative in sorted(
        {item["resolved"] for item in bibliography_commands if item["resolved"] is not None}
    ):
        path = root / relative
        keys = _bibliography_keys(path.read_text(encoding="utf-8"))
        all_bibliography_keys.extend(keys)
        bibliography_files.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "entry_count": len(keys),
                "duplicate_keys": sorted(key for key, count in Counter(keys).items() if count > 1),
            }
        )
    cited_keys = {item["key"] for item in citations if item["key"]}
    bibliography_key_set = set(all_bibliography_keys)
    missing_citations = sorted(cited_keys - bibliography_key_set)
    uncited_bibliography_entries = sorted(bibliography_key_set - cited_keys)

    source_files = [
        {
            "path": relative,
            "sha256": sha256_bytes(source.raw_text.encode("utf-8")),
            "line_count": len(source.raw_text.splitlines()),
        }
        for relative, source in sorted(sources.items())
    ]
    referenced_graphics = sorted(
        {item["resolved"] for item in graphics if item["resolved"] is not None}
    )
    graphic_files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in GRAPHIC_EXTENSIONS
    )
    artifact_records = [*source_files, *bibliography_files]
    for relative in referenced_graphics:
        artifact_records.append({"path": relative, "sha256": sha256_file(root / relative)})

    section_titles = [item["title"].casefold() for item in graph["section_order"]]
    availability = {
        "data_availability": [
            item
            for item in graph["section_order"]
            if item["title"].casefold() == "data availability"
        ],
        "code_availability": [
            item
            for item in graph["section_order"]
            if item["title"].casefold() == "code availability"
        ],
        "methods_present": "methods" in section_titles,
    }

    figure_display_items = _assign_display_ids(
        _flatten_display_items(figures), "DISPLAY-MAIN-FIG"
    )
    table_display_items = _assign_display_ids(
        _flatten_display_items(tables), "DISPLAY-MAIN-TABLE"
    )
    display_items = [*figure_display_items, *table_display_items]

    attention_reasons: list[str] = []
    if graph["missing_inputs"]:
        attention_reasons.append("missing_input")
    if graph["include_cycles"]:
        attention_reasons.append("include_cycle")
    if any(item["resolved"] is None for item in graphics):
        attention_reasons.append("missing_graphic")
    if any(item["resolved"] is None for item in bibliography_commands):
        attention_reasons.append("missing_bibliography")
    if duplicate_labels:
        attention_reasons.append("duplicate_label")
    if missing_references:
        attention_reasons.append("unresolved_reference")
    if missing_citations:
        attention_reasons.append("unresolved_citation")
    if not availability["data_availability"]:
        attention_reasons.append("missing_data_availability")
    if not availability["code_availability"]:
        attention_reasons.append("missing_code_availability")
    if sober_hits:
        attention_reasons.append("editorial_language_review")
    if placeholder_hits:
        attention_reasons.append("placeholder_marker")

    return {
        "entry": entry.relative_to(root).as_posix(),
        "source_fingerprint_sha256": _fingerprint(artifact_records),
        "source_files": source_files,
        "document_graph": graph,
        "figures": figures,
        "tables": tables,
        "display_items": {
            "figures": figure_display_items,
            "tables": table_display_items,
            "all": display_items,
        },
        "graphics": {
            "references": graphics,
            "referenced_files": referenced_graphics,
            "missing_targets": sorted(
                {item["target"] for item in graphics if item["resolved"] is None}
            ),
            "unreferenced_files": sorted(set(graphic_files) - set(referenced_graphics)),
        },
        "labels": {
            "definitions": labels,
            "references": references,
            "duplicate_keys": duplicate_labels,
            "missing_reference_targets": missing_references,
            "unreferenced_keys": unreferenced_labels,
        },
        "bibliography": {
            "commands": bibliography_commands,
            "files": bibliography_files,
            "citations": citations,
            "cited_key_count": len(cited_keys),
            "missing_citation_keys": missing_citations,
            "uncited_entry_keys": uncited_bibliography_entries,
        },
        "availability": availability,
        "sober_language": {
            "hits": sorted(
                sober_hits, key=lambda item: (item["file"], item["line"], item["rule_id"])
            ),
            "counts_by_category": dict(
                sorted(Counter(item["category"] for item in sober_hits).items())
            ),
            "counts_by_rule": dict(
                sorted(Counter(item["rule_id"] for item in sober_hits).items())
            ),
        },
        "placeholders": {
            "hits": placeholder_hits,
            "active_count": sum(1 for item in placeholder_hits if item["active"]),
            "comment_count": sum(1 for item in placeholder_hits if not item["active"]),
        },
        "summary": {
            "status": "attention_required" if attention_reasons else "passed",
            "attention_reasons": attention_reasons,
            "source_file_count": len(source_files),
            "section_count": len(graph["section_order"]),
            "figure_count": len(figures),
            "table_count": len(tables),
            "figure_display_item_count": len(figure_display_items),
            "table_display_item_count": len(table_display_items),
            "display_item_count": len(display_items),
            "label_count": len(labels),
            "reference_command_count": reference_command_count,
            "referenced_key_count": len(referenced_keys),
            "citation_command_count": citation_command_count,
            "cited_key_count": len(cited_keys),
            "sober_language_hit_count": len(sober_hits),
            "placeholder_hit_count": len(placeholder_hits),
        },
    }


def _zip_path_is_safe(name: str) -> bool:
    if "\\" in name or name.startswith("/"):
        return False
    path = PurePosixPath(name)
    return all(part not in {"", ".", ".."} for part in path.parts)


def _resolve_zip_target(
    names: set[str], source_name: str, target: str, extensions: Sequence[str]
) -> str | None:
    clean_target = collapse_whitespace(target).strip()
    if not clean_target or "\\" in clean_target or clean_target.startswith("/"):
        return None
    target_path = PurePosixPath(clean_target)
    suffixes = ("",) if target_path.suffix else tuple(extensions)
    source_parent = PurePosixPath(source_name).parent
    bases = (PurePosixPath(), source_parent)
    for base in bases:
        for suffix in suffixes:
            candidate = (base / f"{clean_target}{suffix}").as_posix()
            if candidate in names:
                return candidate
    return None


def audit_supplementary_zip(path: Path) -> dict[str, Any]:
    path = path.resolve()
    entries: list[dict[str, Any]] = []
    tex_files: list[dict[str, Any]] = []
    all_graphics: list[dict[str, Any]] = []
    all_inputs: list[dict[str, Any]] = []
    all_figures: list[dict[str, Any]] = []
    all_tables: list[dict[str, Any]] = []
    placeholder_hits: list[dict[str, Any]] = []
    warnings: list[str] = []

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        name_set = set(names)
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        unsafe_names = sorted(name for name in names if not _zip_path_is_safe(name))
        bad_crc_entry = archive.testzip()
        for info in infos:
            payload = b"" if info.is_dir() else archive.read(info)
            suffix = PurePosixPath(info.filename).suffix.lower()
            entries.append(
                {
                    "path": info.filename,
                    "size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "sha256": sha256_bytes(payload),
                    "encrypted": bool(info.flag_bits & 0x1),
                }
            )
            if suffix != ".tex" or info.is_dir():
                continue
            text = payload.decode("utf-8", errors="replace")
            source = _source_from_text(info.filename, text)
            active = source.active_text
            placeholder_hits.extend(_placeholder_hits(source))
            sections = [
                _plain_latex(command.argument)
                for command in iter_commands(active, SECTION_COMMANDS)
            ]
            graphics = []
            for command in iter_commands(active, ("includegraphics",)):
                resolved = _resolve_zip_target(
                    name_set, info.filename, command.argument, GRAPHIC_EXTENSIONS
                )
                record = {
                    "source": info.filename,
                    "target": collapse_whitespace(command.argument),
                    "resolved": resolved,
                }
                graphics.append(record)
                all_graphics.append(record)
            inputs = []
            for command in iter_commands(active, ("input", "include")):
                resolved = _resolve_zip_target(name_set, info.filename, command.argument, (".tex",))
                record = {
                    "source": info.filename,
                    "target": collapse_whitespace(command.argument),
                    "resolved": resolved,
                }
                inputs.append(record)
                all_inputs.append(record)
            figures = _find_environments(source, "figure")
            for figure in figures:
                figure["resolved_graphics"] = [
                    _resolve_zip_target(name_set, info.filename, target, GRAPHIC_EXTENSIONS)
                    for target in figure["graphics"]
                ]
                for display_item in figure["display_items"]:
                    display_item["resolved_graphics"] = [
                        _resolve_zip_target(
                            name_set, info.filename, target, GRAPHIC_EXTENSIONS
                        )
                        for target in display_item["graphics"]
                    ]
            tables = _find_environments(source, "table")
            all_figures.extend(figures)
            all_tables.extend(tables)
            tex_files.append(
                {
                    "path": info.filename,
                    "documentclass_present": bool(re.search(r"\\documentclass", active)),
                    "document_environment_present": bool(
                        re.search(r"\\begin\s*\{document\}", active)
                    ),
                    "section_titles": sections,
                    "figure_count": len(figures),
                    "table_count": len(tables),
                    "figures": figures,
                    "tables": tables,
                    "graphics": graphics,
                    "inputs": inputs,
                }
            )

    all_figures.sort(key=lambda item: (item["file"], item["line"], item["index_in_file"]))
    all_tables.sort(key=lambda item: (item["file"], item["line"], item["index_in_file"]))
    figure_display_items = _assign_display_ids(
        _flatten_display_items(all_figures), "DISPLAY-SUPP-FIG"
    )
    table_display_items = _assign_display_ids(
        _flatten_display_items(all_tables), "DISPLAY-SUPP-TABLE"
    )
    display_items = [*figure_display_items, *table_display_items]

    main_candidates = sorted(
        item["path"]
        for item in tex_files
        if item["documentclass_present"] and item["document_environment_present"]
    )
    if any(PurePosixPath(name).name.casefold() == "mian.tex" for name in main_candidates):
        warnings.append("probable_main_filename_typo:mian.tex")
    if not main_candidates:
        warnings.append("no_standalone_tex_entry_detected")
    if bad_crc_entry:
        warnings.append(f"crc_failure:{bad_crc_entry}")
    if unsafe_names:
        warnings.append("unsafe_archive_path")
    if duplicate_names:
        warnings.append("duplicate_archive_entry")
    if any(item["resolved"] is None for item in all_graphics):
        warnings.append("missing_graphic_in_archive")
    if any(item["resolved"] is None for item in all_inputs):
        warnings.append("missing_input_in_archive")
    if placeholder_hits:
        warnings.append("placeholder_marker_in_tex")

    extension_counts = Counter(
        PurePosixPath(item["path"]).suffix.lower() or "<none>" for item in entries
    )
    referenced_graphics = sorted(
        {item["resolved"] for item in all_graphics if item["resolved"] is not None}
    )
    graphic_entries = sorted(
        item["path"]
        for item in entries
        if PurePosixPath(item["path"]).suffix.lower() in GRAPHIC_EXTENSIONS
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "entry_count": len(entries),
        "extension_counts": dict(sorted(extension_counts.items())),
        "bad_crc_entry": bad_crc_entry,
        "unsafe_names": unsafe_names,
        "duplicate_names": duplicate_names,
        "encrypted_entries": sorted(item["path"] for item in entries if item["encrypted"]),
        "main_candidates": main_candidates,
        "tex_files": tex_files,
        "figures": all_figures,
        "tables": all_tables,
        "display_items": {
            "figures": figure_display_items,
            "tables": table_display_items,
            "all": display_items,
        },
        "display_item_count": len(display_items),
        "placeholders": {
            "hits": placeholder_hits,
            "active_count": sum(1 for item in placeholder_hits if item["active"]),
            "comment_count": sum(1 for item in placeholder_hits if not item["active"]),
        },
        "referenced_graphics": referenced_graphics,
        "unreferenced_graphics": sorted(set(graphic_entries) - set(referenced_graphics)),
        "missing_graphics": [item for item in all_graphics if item["resolved"] is None],
        "missing_inputs": [item for item in all_inputs if item["resolved"] is None],
        "warnings": warnings,
        "entries": entries,
        "status": "attention_required" if warnings else "passed",
    }


def build_audit(
    manuscript_root: Path,
    entry: str,
    supplementary: Path | None = None,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "manuscript": audit_manuscript(manuscript_root, entry),
    }
    if supplementary is not None:
        audit["supplementary"] = audit_supplementary_zip(supplementary)
    return audit


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript-root", type=Path, default=Path("paper/manuscript"))
    parser.add_argument("--entry", default="sn-article.tex")
    parser.add_argument(
        "--supplementary",
        type=Path,
        default=Path("paper/manuscript/supplementary/Supplementary_Information.zip"),
    )
    parser.add_argument("--no-supplementary", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    supplementary = None if args.no_supplementary else args.supplementary
    try:
        audit = build_audit(args.manuscript_root, args.entry, supplementary)
    except (FileNotFoundError, OSError, UnicodeError, zipfile.BadZipFile) as error:
        print(f"manuscript audit failed: {error}", file=sys.stderr)
        return 2
    serialized = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        summary = audit["manuscript"]["summary"]
        print(
            f"wrote {args.output}: status={summary['status']} "
            f"figures={summary['figure_count']} tables={summary['table_count']} "
            f"sober_hits={summary['sober_language_hit_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
