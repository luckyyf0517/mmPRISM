from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    script = Path(__file__).parents[2] / "paper/manager/tools/audit_manuscript.py"
    spec = importlib.util.spec_from_file_location("mmprism_manuscript_audit", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def test_strip_latex_comments_preserves_escaped_percent_and_offsets() -> None:
    source = "active 84\\% text % hidden novel\n% full line\nnext\n"
    stripped = AUDIT.strip_latex_comments(source)

    assert len(stripped) == len(source)
    assert stripped.count("\n") == source.count("\n")
    assert "84\\% text" in stripped
    assert "hidden novel" not in stripped
    assert "next" in stripped


def test_build_audit_follows_active_inputs_and_checks_contracts(tmp_path: Path) -> None:
    root = tmp_path / "manuscript"
    (root / "chapter").mkdir(parents=True)
    (root / "pics").mkdir()
    (root / "pics/figure.pdf").write_bytes(b"figure")
    (root / "pics/figure2.pdf").write_bytes(b"figure2")
    (root / "refs.bib").write_text(
        "@article{known,\n title={Known}\n}\n",
        encoding="utf-8",
    )
    (root / "main.tex").write_text(
        r"""
\documentclass{article}
\begin{document}
\section{Methods}\label{sec:methods}
\input{chapter/results}
% \section{Data Availability}
\bibliography{refs}
\end{document}
""",
        encoding="utf-8",
    )
    (root / "chapter/results.tex").write_text(
        r"""
\section{Results}
This novel result cites \cite{known,missing} and references \ref{fig:one}.
% superior commented claim
\begin{figure}
\includegraphics[width=1.0\linewidth]{pics/figure}
\caption{An active caption.}
\label{fig:one}
\includegraphics{pics/figure2}
\caption{A second caption in the same environment.}
\label{fig:two}
\end{figure}
""",
        encoding="utf-8",
    )

    audit = AUDIT.build_audit(root, "main.tex")
    manuscript = audit["manuscript"]

    assert [item["path"] for item in manuscript["source_files"]] == [
        "chapter/results.tex",
        "main.tex",
    ]
    assert [item["title"] for item in manuscript["document_graph"]["section_order"]] == [
        "Methods",
        "Results",
    ]
    assert manuscript["summary"]["figure_count"] == 1
    assert manuscript["summary"]["figure_display_item_count"] == 2
    assert manuscript["summary"]["display_item_count"] == 2
    assert manuscript["summary"]["table_count"] == 0
    assert manuscript["summary"]["citation_command_count"] == 1
    assert manuscript["summary"]["reference_command_count"] == 1
    assert manuscript["graphics"]["missing_targets"] == []
    assert manuscript["labels"]["missing_reference_targets"] == []
    assert manuscript["bibliography"]["missing_citation_keys"] == ["missing"]
    assert manuscript["availability"]["data_availability"] == []
    figure_items = manuscript["display_items"]["figures"]
    assert [item["display_id"] for item in figure_items] == [
        "DISPLAY-MAIN-FIG-01",
        "DISPLAY-MAIN-FIG-02",
    ]
    assert figure_items[0]["resolved_graphics"] == ["pics/figure.pdf"]
    assert figure_items[1]["resolved_graphics"] == ["pics/figure2.pdf"]
    sober_contexts = [item["context"] for item in manuscript["sober_language"]["hits"]]
    assert any("novel result" in context for context in sober_contexts)
    assert all("commented claim" not in context for context in sober_contexts)


def test_supplementary_zip_audit_checks_crc_assets_and_probable_typo(tmp_path: Path) -> None:
    archive_path = tmp_path / "supplement.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "mian.tex",
            r"""\documentclass{article}
\begin{document}
\section{Supplementary Note 1}
\begin{figure}
\includegraphics{pics/result.pdf}
\caption{A supplementary result.}
\label{fig:s1}
\end{figure}
\begin{table}
\caption{A supplementary table.}
\label{tab:s1}
% 示例表格内容（替换为真实数据）
\end{table}
\end{document}
""",
        )
        archive.writestr("pics/result.pdf", b"pdf")

    result = AUDIT.audit_supplementary_zip(archive_path)

    assert result["bad_crc_entry"] is None
    assert result["main_candidates"] == ["mian.tex"]
    assert result["missing_graphics"] == []
    assert result["referenced_graphics"] == ["pics/result.pdf"]
    assert result["unreferenced_graphics"] == []
    figure = result["display_items"]["figures"][0]
    table = result["display_items"]["tables"][0]
    assert figure["display_id"] == "DISPLAY-SUPP-FIG-01"
    assert figure["labels"] == ["fig:s1"]
    assert figure["resolved_graphics"] == ["pics/result.pdf"]
    assert figure["section"] == "Supplementary Note 1"
    assert table["display_id"] == "DISPLAY-SUPP-TABLE-01"
    assert table["labels"] == ["tab:s1"]
    assert result["display_item_count"] == 2
    assert result["placeholders"]["active_count"] == 0
    assert result["placeholders"]["comment_count"] == 1
    assert result["warnings"] == [
        "probable_main_filename_typo:mian.tex",
        "placeholder_marker_in_tex",
    ]
    assert result["status"] == "attention_required"
