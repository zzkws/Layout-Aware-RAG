"""Build four GitHub Release archives without putting large data in Git history."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from tools.public_data import public_source_name


def sanitize_payload(value: object) -> object:
    """Remove machine-local source paths while preserving the JSON structure."""
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: public_source_name(str(item))
            if key == "source_pdf"
            else sanitize_payload(item)
            for key, item in value.items()
        }
    return value


def sanitized_json_text(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8")
    if '"source_pdf"' not in text:
        return None
    if path.suffix == ".jsonl":
        rows = [sanitize_payload(json.loads(line)) for line in text.splitlines() if line]
        return "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        ) + "\n"
    payload = sanitize_payload(json.loads(text))
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def add_tree(
    archive: zipfile.ZipFile,
    root: Path,
    prefix: str,
    sanitize_index: bool = False,
) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        target = f"{prefix}/{relative}"
        sanitized = (
            sanitized_json_text(path)
            if sanitize_index and path.suffix in {".json", ".jsonl"}
            else None
        )
        if sanitized is not None:
            archive.writestr(target, sanitized)
            continue
        archive.write(path, target)


def make_archive(source: Path, output: Path, prefix: str, sanitize_index: bool = False) -> None:
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        add_tree(archive, source, prefix, sanitize_index=sanitize_index)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--txc-pdf", type=Path, required=True)
    parser.add_argument("--txc-data", type=Path, required=True)
    parser.add_argument("--tkd-pdf", type=Path, required=True)
    parser.add_argument("--tkd-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("release_artifacts"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = [
        (args.txc_pdf, args.out / "txc-source-pdfs-v0.1.0.zip", "corpora/txc/pdf", False),
        (args.txc_data, args.out / "txc-derived-v0.1.0.zip", "corpora/txc/data", True),
        (args.tkd_pdf, args.out / "tkd-source-pdfs-v0.1.0.zip", "corpora/tkd/pdf", False),
        (args.tkd_data, args.out / "tkd-derived-v0.1.0.zip", "corpora/tkd/data", True),
    ]
    checksums = []
    for source, output, prefix, sanitize in specs:
        if not source.is_dir():
            raise SystemExit(f"missing source directory: {source}")
        print(f"packaging {output.name} ...", flush=True)
        make_archive(source.resolve(), output.resolve(), prefix, sanitize_index=sanitize)
        checksums.append(f"{sha256(output)}  {output.name}")

    checksum_file = args.out / "SHA256SUMS.txt"
    checksum_file.write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(checksum_file)


if __name__ == "__main__":
    main()
