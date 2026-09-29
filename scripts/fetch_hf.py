#!/usr/bin/env python3
"""Resumable Hugging Face downloader (models and datasets) for slow or unstable networks.

Downloads repo files with curl (resume + retries), trying a list of endpoints in order, and verifies every
LFS file against the sha256 published by the official huggingface.co API, so a mirror cannot silently alter
the content. Models go to models/<org>--<name>/ (picked up by adas-vla automatically).

  python scripts/fetch_hf.py Qwen/Qwen2.5-VL-3B-Instruct
  python scripts/fetch_hf.py --dataset commaai/comma2k19 --include "raw_data/Chunk_1.zip" --out data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model"]
DEFAULT_ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]
OFFICIAL = "https://huggingface.co"


REQUIRE_SSID: str | None = None  # set by --require-ssid: only download while on this WiFi network


def current_ssid() -> str | None:
    out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:]
    return None


def on_allowed_network() -> bool:
    return REQUIRE_SSID is None or current_ssid() == REQUIRE_SSID


def wait_for_allowed_network() -> None:
    if on_allowed_network():
        return
    print(f"   paused: not on WiFi '{REQUIRE_SSID}' (on '{current_ssid()}'), waiting...", flush=True)
    while not on_allowed_network():
        time.sleep(15)
    print(f"   resumed on '{REQUIRE_SSID}'", flush=True)


def run_guarded(cmd: list[str]) -> int:
    """Run curl, killing it at once if the machine leaves the allowed network (e.g. falls back to a
    metered phone hotspot)."""
    proc = subprocess.Popen(cmd)
    while proc.poll() is None:
        time.sleep(2)
        if not on_allowed_network():
            proc.terminate()
            proc.wait()
            return 99
    return proc.returncode


def hf_token() -> str | None:
    """Token from HF_TOKEN or `hf auth login`; only ever sent to huggingface.co, never to a mirror."""
    import os

    token = os.environ.get("HF_TOKEN")
    path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text().strip()
    return token or None


def list_files(repo: str, repo_type: str = "model", patterns: list[str] | None = None,
               attempts: int = 30) -> list[dict]:
    url = f"https://huggingface.co/api/{repo_type}s/{repo}?blobs=true"
    token = hf_token()
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                data = json.load(r)
            break
        except OSError as e:  # URLError, DNS failure, timeout
            if attempt == attempts:
                raise
            print(f"   API unreachable ({e}), retry in 30 s", flush=True)
            time.sleep(30)
    files = []
    for s in data["siblings"]:
        name = s["rfilename"]
        if patterns:
            if not any(fnmatch(name, p) for p in patterns):
                continue
        elif "/" in name or not any(fnmatch(name, p) for p in PATTERNS):
            continue
        files.append({"name": name, "size": s.get("size"), "sha256": (s.get("lfs") or {}).get("sha256")})
    return files


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def is_complete(path: Path, info: dict) -> bool:
    if not path.exists() or (info["size"] is not None and path.stat().st_size != info["size"]):
        return False
    return info["sha256"] is None or sha256_of(path) == info["sha256"]


def download_resumable(repo: str, info: dict, part: Path, endpoints: list[str], attempts: int = 2000,
                       repo_type: str = "model") -> bool:
    """Resume from the current .part size on every attempt.

    The retry loop lives here, not in curl: curl's own --retry can restart a stalled transfer from zero.
    Stalls are cut after 30 s below 20 KB/s, endpoints rotate after 3 attempts without progress, and the
    wait grows (up to 60 s) while the network is down so a long outage does not burn through all attempts.
    """
    idx, stuck, no_progress = 0, 0, 0
    for attempt in range(1, attempts + 1):
        wait_for_allowed_network()
        before = part.stat().st_size if part.exists() else 0
        if info["size"] is not None and before >= info["size"]:
            return True
        prefix = "datasets/" if repo_type == "dataset" else ""
        url = f"{endpoints[idx]}/{prefix}{repo}/resolve/main/{urllib.parse.quote(info['name'])}"
        cmd = ["curl", "-fL", "--connect-timeout", "20", "--speed-limit", "20000", "--speed-time", "30",
               "-C", "-", "-o", str(part), url]
        token = hf_token()
        if token and endpoints[idx] == OFFICIAL:
            # Token via a header file, so it never shows up in the process list; curl drops it on
            # cross-host redirects (to the CDN), which carry their own signed URLs.
            header_file = part.with_name(".auth_header")
            header_file.write_text(f"Authorization: Bearer {token}\n")
            header_file.chmod(0o600)
            cmd[1:1] = ["-H", f"@{header_file}"]
        cmd.insert(1, "--progress-bar" if sys.stdout.isatty() else "-sS")
        try:
            rc = run_guarded(cmd)
        finally:
            part.with_name(".auth_header").unlink(missing_ok=True)  # never leave the token on disk
        after = part.stat().st_size if part.exists() else 0
        if rc == 0:
            return True
        stuck = 0 if after > before else stuck + 1
        if stuck >= 3:
            idx, stuck = (idx + 1) % len(endpoints), 0
        print(f"   retry {attempt}: {after / 1e6:.0f} MB so far (curl exit {rc}), next: {endpoints[idx]}",
              flush=True)
        time.sleep(3 if after > before else min(60, 5 * (1 + no_progress)))
        no_progress = 0 if after > before else no_progress + 1
    return False


def fetch(repo: str, out_root: Path, endpoints: list[str], repo_type: str = "model",
          patterns: list[str] | None = None) -> bool:
    out_dir = out_root / repo.replace("/", "--")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_files(repo, repo_type, patterns)
    total = sum(f["size"] or 0 for f in files)
    print(f"== {repo}: {len(files)} files, {total / 1e9:.2f} GB -> {out_dir}", flush=True)
    for info in files:
        target = out_dir / info["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if is_complete(target, info):
            continue
        part = target.with_name(target.name + ".part")
        print(f"   {info['name']} ({(info['size'] or 0) / 1e6:.1f} MB)", flush=True)
        if not download_resumable(repo, info, part, endpoints, repo_type=repo_type):
            print(f"!! failed: {info['name']}", flush=True)
            return False
        if info["sha256"] and sha256_of(part) != info["sha256"]:
            print(f"!! sha256 mismatch for {info['name']}, deleting partial file", flush=True)
            part.unlink()
            return False
        part.rename(target)
    print(f"== done {repo}", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "models"))
    ap.add_argument("--endpoint", action="append", help="download endpoint(s), tried in order")
    ap.add_argument("--dataset", action="store_true", help="repos are datasets, not models")
    ap.add_argument("--include", action="append", help="glob(s) on repo paths (default: model weights/config)")
    ap.add_argument("--gated", action="store_true", help="gated repo: download only from huggingface.co with your token")
    ap.add_argument("--require-ssid", help="only download while connected to this WiFi (pause otherwise)")
    args = ap.parse_args()
    global REQUIRE_SSID
    REQUIRE_SSID = args.require_ssid
    if args.gated and not args.endpoint:
        args.endpoint = [OFFICIAL]
    repo_type = "dataset" if args.dataset else "model"
    wait_for_allowed_network()
    ok = all([fetch(r, Path(args.out), args.endpoint or DEFAULT_ENDPOINTS, repo_type, args.include)
              for r in args.repos])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
