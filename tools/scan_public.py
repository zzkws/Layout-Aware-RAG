"""Fail when publishable text contains credentials, private hosts, or machine paths."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_PARTS = {
    ".git",
    ".next",
    ".venv",
    ".vinext",
    ".wrangler",
    "data",
    "data_tkd",
    "dist",
    "models",
    "node_modules",
    "release_artifacts",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".mjs",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_SECRET_NAMES = {".deepseek_key", ".gemini_key", ".env"}
RULES = {
    "private IPv4 address": re.compile(
        r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2}\b"
    ),
    "private host name": re.compile(r"\btxc-server\b", re.IGNORECASE),
    "Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "Google-style API key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-)?[0-9A-Za-z_-]{20,}\b"),
}


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in LOCAL_SECRET_NAMES or path.name.startswith(".env."):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in RULES.items():
                if pattern.search(line):
                    findings.append(f"{path.relative_to(root)}:{line_no}: {label}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    findings = scan(args.root.resolve())
    if findings:
        raise SystemExit("Public-safety scan failed:\n" + "\n".join(findings))
    print("Public-safety scan passed.")


if __name__ == "__main__":
    main()
