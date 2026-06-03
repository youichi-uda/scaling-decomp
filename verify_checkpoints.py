#!/usr/bin/env python3
"""Verify Pythia intermediate-step checkpoint integrity on the HuggingFace Hub.

For each (repo, file, revision) triple, query the HF Hub resolve endpoint
HEAD and read the x-linked-etag (= LFS blob SHA256) and x-linked-size headers.
These are server-side metadata — they reflect what blob the Hub itself maps
the revision to, independent of the transformers / huggingface_hub client
resolution layer.

This decisively pins the bug locus described in Appendix C of the paper:
- For Pythia-2.8B model.safetensors at step1000..step100000, the
  x-linked-etag matches main's, proving the Hub itself maps those revisions
  to the main-weights blob (not a client-side resolution error).
- For Pythia-12B pytorch_model-00003-of-00003.bin at step50000, the
  x-linked-size is 3.9 GB (vs 4.1 GB at neighbouring revisions), proving the
  Hub blob itself is truncated.

Usage:
  python3 verify_checkpoints.py
  python3 verify_checkpoints.py --json > hub_blob_audit.json
"""
from __future__ import annotations
import argparse, json, sys
from typing import Any
from urllib.request import Request, urlopen, HTTPRedirectHandler, build_opener


class _NoRedirect(HTTPRedirectHandler):
    """The x-linked-etag header sits on the Hub's resolve response BEFORE the
    CDN redirect. Following the redirect to the actual blob URL strips it, so
    we disable redirect-following here."""

    def http_error_301(self, req, fp, code, msg, headers):
        return fp

    def http_error_302(self, req, fp, code, msg, headers):
        return fp

    http_error_303 = http_error_302
    http_error_307 = http_error_302
    http_error_308 = http_error_302


_opener = build_opener(_NoRedirect)

CHECKS = [
    # (repo, filename, [revisions])
    ("EleutherAI/pythia-2.8b", "model.safetensors",
        ["main", "step1000", "step2000", "step4000", "step10000",
         "step50000", "step100000", "step143000"]),
    ("EleutherAI/pythia-2.8b", "pytorch_model.bin",
        ["main", "step1000", "step2000", "step10000",
         "step50000", "step100000", "step143000"]),
    ("EleutherAI/pythia-12b", "pytorch_model-00003-of-00003.bin",
        ["main", "step42000", "step50000", "step70000", "step100000"]),
]


def head(url: str) -> dict[str, str]:
    req = Request(url, method="HEAD")
    with _opener.open(req, timeout=30) as resp:
        return {k.lower(): v for k, v in resp.headers.items()}


def fetch_revision_info(repo: str, fname: str, rev: str) -> dict[str, Any]:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{fname}"
    h = head(url)
    return {
        "repo": repo,
        "file": fname,
        "revision": rev,
        "commit_sha": h.get("x-repo-commit"),
        "blob_sha256": (h.get("x-linked-etag") or "").strip('"'),
        "blob_size": int(h.get("x-linked-size", 0)) or None,
    }


def emit_text(rows: list[dict[str, Any]]) -> None:
    by_repo_file: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_repo_file.setdefault((r["repo"], r["file"]), []).append(r)
    for (repo, fname), entries in by_repo_file.items():
        print(f"\n=== {repo} :: {fname} ===")
        print(f"{'revision':<12} {'blob SHA256[:16]':<18} {'blob size':<14} {'commit SHA[:12]':<14} note")
        main_blob = next((e["blob_sha256"] for e in entries if e["revision"] == "main"), None)
        main_size = next((e["blob_size"] for e in entries if e["revision"] == "main"), None)
        for e in entries:
            blob = (e["blob_sha256"] or "<none>")[:16]
            size = f"{e['blob_size']:>12,}" if e["blob_size"] else "<none>"
            commit = (e["commit_sha"] or "<none>")[:12]
            note = ""
            if e["revision"] != "main":
                if main_blob and e["blob_sha256"] == main_blob:
                    note = "** blob == main (MIS-REGISTRATION) **"
                elif main_size and e["blob_size"] and e["blob_size"] != main_size:
                    pct = 100.0 * (e["blob_size"] - main_size) / main_size
                    note = f"size delta vs main: {pct:+.1f}% (truncation candidate)" if pct < -1.0 else f"size delta {pct:+.1f}%"
            print(f"{e['revision']:<12} {blob:<18} {size:<14} {commit:<14} {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    rows = []
    for repo, fname, revs in CHECKS:
        for rev in revs:
            try:
                rows.append(fetch_revision_info(repo, fname, rev))
            except Exception as e:
                rows.append({"repo": repo, "file": fname, "revision": rev,
                             "error": f"{type(e).__name__}: {e}"})

    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        emit_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
