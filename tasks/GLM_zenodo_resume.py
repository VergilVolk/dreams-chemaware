"""Robust resumable multi-connection downloader for Zenodo (or any Range-supporting server).

Design (learned from field experience 20261006-07):
- Zenodo throttles single connections to ~16KB/s and rejects >~10 concurrent.
- 6 workers x small chunks (8MB) with per-chunk resume is the sweet spot.
- Each chunk retries indefinitely with backoff; partial chunk bytes are kept
  and the next attempt resumes mid-chunk via a computed sub-range.
- Parts concatenate onto the main file only when every chunk is complete.

Usage:
  python -X utf8 tasks/GLM_zenodo_resume.py --url URL --output PATH --size N
"""
import argparse
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 8 * 1024 * 1024  # 8MB chunks
WORKERS = 6
ATTEMPT_TIMEOUT = 120  # seconds per HTTP attempt
BACKOFF_MAX = 60  # seconds


def fetch_range(url: str, start: int, end: int, out: Path, state: dict):
    """Fetch bytes [start, end] inclusive, appending to `out`; resume-safe."""
    have = out.stat().st_size if out.exists() else 0
    want = end - start + 1
    backoff = 2.0
    while have < want:
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start + have}-{end}",
                              "User-Agent": "Mozilla/5.0 (research-download)"})
            with urllib.request.urlopen(req, timeout=ATTEMPT_TIMEOUT) as r:
                with open(out, "ab") as f:
                    while True:
                        buf = r.read(1 << 17)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        state["bytes"] += len(buf)
            backoff = 2.0
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError, OSError) as e:
            state["errors"] += 1
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff = min(backoff * 1.7, BACKOFF_MAX)
            have = out.stat().st_size if out.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--size", type=int, required=True)
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.stat().st_size if out.exists() else 0
    if have >= args.size:
        print(f"already complete: {have} bytes", flush=True)
        return
    print(f"main file has {have:,} / {args.size:,}; downloading the rest",
          flush=True)

    parts_dir = out.parent / (out.name + ".parts")
    parts_dir.mkdir(exist_ok=True)

    # Build chunk list over the remaining range [have, size)
    chunks = []
    s = have
    while s < args.size:
        e = min(s + CHUNK - 1, args.size - 1)
        idx = len(chunks)
        chunks.append((s, e, parts_dir / f"part{idx:05d}.bin"))
        s = e + 1

    q: "queue.Queue[tuple[int, int, Path]]" = queue.Queue()
    for c in chunks:
        want = c[1] - c[0] + 1
        got = c[2].stat().st_size if c[2].exists() else 0
        if got < want:
            q.put(c)
    state = {"bytes": 0, "errors": 0}
    t0 = time.time()

    def progress():
        while not progress.stop.is_set():
            time.sleep(30)
            done = sum(1 for c in chunks if c[2].exists()
                       and c[2].stat().st_size == c[1] - c[0] + 1)
            rate = state["bytes"] / max(1e-9, time.time() - t0)
            remain_bytes = sum((c[1] - c[0] + 1) -
                               (c[2].stat().st_size if c[2].exists() else 0)
                               for c in chunks)
            eta_h = remain_bytes / rate / 3600 if rate > 1 else float("inf")
            print(f"[{time.strftime('%H:%M:%S')}] chunks {done}/{len(chunks)} "
                  f"rate {rate/1024:.0f}KB/s remain {remain_bytes/1e6:.0f}MB "
                  f"eta {eta_h:.2f}h errors {state['errors']}", flush=True)

    progress.stop = threading.Event()  # type: ignore[attr-defined]
    pt = threading.Thread(target=progress, daemon=True)
    pt.start()

    def worker():
        while True:
            try:
                s, e, p = q.get_nowait()
            except queue.Empty:
                return
            fetch_range(args.url, s, e, p, state)
            q.task_done()

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    progress.stop.set()  # type: ignore[attr-defined]

    # Verify all chunks complete, then concatenate
    for s, e, p in chunks:
        want = e - s + 1
        got = p.stat().st_size if p.exists() else 0
        if got != want:
            print(f"FATAL: part {p.name} has {got:,} / {want:,}", flush=True)
            sys.exit(1)
    with open(out, "ab") as f:
        for _, _, p in chunks:
            f.write(p.read_bytes())
    final = out.stat().st_size
    print(f"final: {final:,} bytes (expected {args.size:,})", flush=True)
    if final == args.size:
        for _, _, p in chunks:
            p.unlink(missing_ok=True)
        parts_dir.rmdir()
        print("parts cleaned; COMPLETE", flush=True)
    else:
        print("SIZE MISMATCH - parts kept for retry", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
