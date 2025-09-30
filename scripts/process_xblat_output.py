#!/usr/bin/env python3
"""Normalize xBLAT logs by stripping carriage returns and extracting pass/fail lines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

FAIL_PATTERN = re.compile(r"\bFAIL(?:ED)?\b", re.IGNORECASE)
FATAL_PATTERN = re.compile(r"FATAL\s+ERROR", re.IGNORECASE)
WARN_PATTERN = re.compile(r"\bWARN(?:ING)?\b", re.IGNORECASE)
PASS_PATTERN = re.compile(r"\bPASS(?:ED)?\b", re.IGNORECASE)


def normalize_text(data: str) -> str:
    # Replace CRLF first, then lone CR, and normalise form-feed to newline.
    data = data.replace("\r\n", "\n")
    data = re.sub(r"\r", "\n", data)
    data = data.replace("\x0c", "\n")
    return data


def extract_markers(lines: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    passes: List[str] = []
    fails: List[str] = []
    warns: List[str] = []

    seen_pass = set()
    seen_fail = set()
    seen_warn = set()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        fail_match = FAIL_PATTERN.search(upper) or FATAL_PATTERN.search(upper)
        warn_match = WARN_PATTERN.search(upper)
        pass_match = PASS_PATTERN.search(upper)

        if fail_match:
            if line not in seen_fail:
                fails.append(line)
                seen_fail.add(line)
            continue
        if warn_match:
            if line not in seen_warn:
                warns.append(line)
                seen_warn.add(line)
            # a warning line might also contain PASS/FAIL text; skip further tagging.
            continue
        if pass_match:
            if line not in seen_pass:
                passes.append(line)
                seen_pass.add(line)
            continue

    return passes, fails, warns


def write_summary(path: Path, passes: List[str], fails: List[str], warns: List[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in passes:
            handle.write(f"PASS\t{line}\n")
        for line in fails:
            handle.write(f"FAIL\t{line}\n")
        for line in warns:
            handle.write(f"WARN\t{line}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize xBLAT output and extract summary markers.")
    parser.add_argument("--input", required=True, help="Raw log file produced by xBLAT")
    parser.add_argument("--normalized", required=True, help="Destination path for normalized text output")
    parser.add_argument("--summary", help="Path to write pass/fail summary (tab-separated)")
    args = parser.parse_args()

    input_path = Path(args.input)
    normalized_path = Path(args.normalized)
    summary_path = Path(args.summary) if args.summary else None

    raw_bytes = input_path.read_bytes()
    if raw_bytes.strip():  # prefer directly captured stdout when available
        source_bytes = raw_bytes
    else:
        source_bytes = raw_bytes
        # Fallback to accompanying *.out files (written by the Fortran harness)
        for candidate in sorted(input_path.parent.glob("*.out")):
            try:
                payload = candidate.read_bytes()
            except OSError:
                continue
            if payload.strip():
                source_bytes = payload
                break

    data = source_bytes.decode("latin-1", errors="replace")
    normalized = normalize_text(data)
    normalized_path.write_text(normalized, encoding="utf-8")

    if summary_path is not None:
        passes, fails, warns = extract_markers(normalized.splitlines())
        write_summary(summary_path, passes, fails, warns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
