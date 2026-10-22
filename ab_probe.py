"""A/B engine builds with the method of tools/ab_engine.py, plus the probes of issue #3252 (scratch tool, not in the repo).

    python ab_probe.py CONFIG ARM ARM [...] --server SRC/serve/server.py --out DIR [--rounds 6] [--probes chat,long,sampled,prefill]
    python ab_probe.py --report DIR            # the summary again from DIR (also on another machine)

ARM = NAME:key=value,...  as in ab_engine.py: exe= the engine, xargs= extra engine args (';'-separated), server= this arm's
serve/server.py (else --server), any other key goes into the server's (and so the engine's) environment.  --body NAME=JSON adds
the JSON object's fields to every request body of arm NAME (e.g. --body pp0='{"presence_penalty": 0}').  Method kept from
ab_engine.py: the server is started once per arm and round, the arms' order alternates over rounds (--order reverse,
ab_engine's; --order rotate gives every arm every position with 3+ arms), one short warm-up answer, then the same requests in
the same order, numbers from the server's `timings` (the engine's clock), medians.

Probes, in this order inside one engine start:
  chat     ab_engine's three short questions, greedy, thinking off, --max-new tokens   (the request body is ab_engine's, byte for byte)
  long     ab_engine's ~6,000-word prompt (~6.3K tokens), the same body
  essay    three greedy answers of --essay-max tokens (about 1,000), so one reply's content matters less
  sampled  two long-answer prompts with the reporter's envelope (temperature 1.0, top_p 0.95, top_k 20, seed 42; no penalty
           field, so the server's own default applies), --sampled-max tokens, thinking off unless --sampled-think
  prefill  a fresh ~23K-token prompt (unique tag first, so no prefix is reused), max_tokens 8, greedy; cache_n must be 0
Additions: cache_n, predicted_n, draft counts, a hash of each reply, ready time, GPU memory before each start, the load average;
after each stop it waits until the GPU memory is back near the start value, so a slow exit cannot shrink the next arm's cache.
--pairs B/A,D/C picks the paired comparisons (default: every arm against the first); each pair also gets the draft acceptance
difference in points, paired by round.  The engine command line each server started is listed per arm, so a server version
that adds an argument of its own shows up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# the same three questions and word list as tools/ab_engine.py (copied, so the file runs on its own)
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)
SAMPLED = (   # stand-ins for the reporter's 'tetris' and 'sand' probes (their exact prompts are not published)
    ("tetris", "Write a complete, playable Tetris game as one HTML file with inline CSS and JavaScript. Return only the file."),
    ("sand", "Write a falling-sand simulation (sand, water, stone, fire) as one HTML file with inline CSS and JavaScript. "
             "Return only the file."),
)


ESSAYS = (    # long greedy answers: about 1,000 decoded tokens per request, so one reply's content matters less
    "Write a detailed technical article of about 900 words that explains how a CPU cache hierarchy works, with headings.",
    "Write a short story of about 900 words about a lighthouse keeper who finds a message in a bottle.",
    "Write a complete Python module of about 150 lines: a small command-line to-do app with argparse, JSON storage and "
    "unit tests.",
)


def long_prompt(words: int, tag: str = "") -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    head = f"{tag}\n" if tag else ""
    return f"{head}Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, body: dict, timeout: float = 3600) -> tuple[dict, float]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), time.monotonic() - t0


def greedy(prompt: str, max_tokens: int) -> dict:          # tools/ab_engine.py's body
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def sampled(prompt: str, max_tokens: int, think: bool) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "seed": 42, "chat_template_kwargs": {"enable_thinking": think}}


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text().split(" ", 3)[:3].__str__()
    except OSError:
        return "?"


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, greedy("Say hi.", 8))
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def wait_gpu_back(base: int | None, slack_mib: int = 512, timeout: float = 120) -> int | None:
    if base is None:
        time.sleep(5)
        return None
    t0, used = time.monotonic(), gpu_used_mib()
    while used is not None and used > base + slack_mib and time.monotonic() - t0 < timeout:
        time.sleep(2)
        used = gpu_used_mib()
    time.sleep(3)
    return used


def row_of(rnd: int, arm: str, tag: str, r: dict, wall: float) -> dict:
    t = r.get("timings") or {}
    msg = (r.get("choices") or [{}])[0].get("message", {})
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    return {"round": rnd, "arm": arm, "req": tag, "decode": t.get("predicted_per_second"), "prompt": t.get("prompt_per_second"),
            "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "predicted_n": t.get("predicted_n"),
            "predicted_ms": t.get("predicted_ms"), "prompt_ms": t.get("prompt_ms"), "draft_n": t.get("draft_n"),
            "draft_ok": t.get("draft_n_accepted"), "usage": r.get("usage"), "finish": (r.get("choices") or [{}])[0].get("finish_reason"),
            "wall_s": round(wall, 2), "sha": hashlib.sha256(text.encode()).hexdigest()[:16], "text": (msg.get("content") or "")[:80]}


def requests_for(a, rnd: int, arm: str, extra: dict | None = None):
    probes = [p.strip() for p in a.probes.split(",") if p.strip()]
    if "chat" in probes:
        for i, p in enumerate(PROMPTS):
            yield f"chat{i}", {**greedy(p, a.max_new), **(extra or {})}
    if "long" in probes:
        yield "long", {**greedy(long_prompt(a.long_words), a.max_new), **(extra or {})}
    if "essay" in probes:
        for i, p in enumerate(ESSAYS):
            yield f"essay{i}", {**greedy(p, a.essay_max), **(extra or {})}
    if "sampled" in probes:
        for tag, p in SAMPLED:
            yield tag, {**sampled(p, a.sampled_max, a.sampled_think), **(extra or {})}
    if "prefill" in probes:
        for words in [int(w) for w in str(a.prefill_words).split(",") if w.strip()]:
            for i in range(a.prefill_n):
                tag = f"[run {rnd}-{arm}-{i}-{uuid.uuid4().hex[:12]}]"     # first in the prompt: nothing before it can match
                name = f"prefill{i}" if "," not in str(a.prefill_words) else f"pf{words // 1000}k{i}"
                yield name, {**greedy(long_prompt(words, tag), 8), **(extra or {})}


def summarize(rows: list[dict], starts: list[dict], arms: list[str], pairs: list[tuple[str, str]] | None = None) -> str:
    out = []
    reqs = list(dict.fromkeys(r["req"] for r in rows))
    out.append("medians over rounds (decode = timings.predicted_per_second, prompt = timings.prompt_per_second):")
    for tag in reqs:
        cells = []
        for name in arms:
            rr = [x for x in rows if x["arm"] == name and x["req"] == tag]
            d = [x["decode"] for x in rr if x["decode"]]
            p = [x["prompt"] for x in rr if x["prompt"]]
            acc = [x["draft_ok"] / x["draft_n"] for x in rr if x.get("draft_n")]
            cells.append(f"{name}: decode {statistics.median(d) if d else float('nan'):6.1f} "
                         f"[{min(d) if d else float('nan'):.1f}-{max(d) if d else float('nan'):.1f}] "
                         f"prompt {statistics.median(p) if p else float('nan'):7.1f} "
                         f"drafts {100 * statistics.median(acc) if acc else float('nan'):4.1f}% n={len(rr)}")
        out.append(f"  {tag:8s} | " + " | ".join(cells))
    if len(arms) >= 2:
        pairs = pairs or [(arms[0], name) for name in arms[1:]]
        out.append("\npaired by round (median of the per-round decode and prompt ratios, rounds in which the arm was faster; "
                   "draft acceptance: median per-round difference in points, rounds in which the arm accepted more):")
        for tag in reqs:
            cells = []
            for ref, name in pairs:
                for metric in ("decode", "prompt", "acc"):
                    vals = []
                    for rnd in sorted({x["round"] for x in rows}):
                        a_ = [x for x in rows if x["arm"] == ref and x["req"] == tag and x["round"] == rnd]
                        b_ = [x for x in rows if x["arm"] == name and x["req"] == tag and x["round"] == rnd]
                        if metric == "acc":
                            a_ = [100 * x["draft_ok"] / x["draft_n"] for x in a_ if x.get("draft_n")]
                            b_ = [100 * x["draft_ok"] / x["draft_n"] for x in b_ if x.get("draft_n")]
                            if a_ and b_:
                                vals.append(statistics.median(b_) - statistics.median(a_))
                        else:
                            a_ = [x[metric] for x in a_ if x[metric]]
                            b_ = [x[metric] for x in b_ if x[metric]]
                            if a_ and b_:
                                vals.append(statistics.median(b_) / statistics.median(a_))
                    if not vals:
                        continue
                    if metric == "acc":
                        cells.append(f"{name}-{ref} acc {statistics.median(vals):+.1f} pt "
                                     f"({sum(1 for q in vals if q > 0)}/{len(vals)})")
                    else:
                        cells.append(f"{name}/{ref} {metric} x{statistics.median(vals):.3f} "
                                     f"({sum(1 for q in vals if q > 1)}/{len(vals)})")
            out.append(f"  {tag:8s} | " + " | ".join(cells))
        same = {}
        for x in rows:
            if x["req"].startswith(("chat", "long", "essay")):
                same.setdefault((x["req"], x["arm"]), set()).add(x["sha"])
        out.append("\ngreedy replies, distinct texts per arm (1 = the same text every round): " +
                   ", ".join(f"{k[0]}/{k[1]}={len(v)}" for k, v in sorted(same.items())))
    out.append("\nstarts: " + "; ".join(f"r{s['round']} {s['arm']} ready {s.get('ready_s', 'FAILED')} s gpu {s.get('gpu_before')}->{s.get('gpu_after')} MiB"
                                         for s in starts))
    return "\n".join(out)


def boot_profile(log: Path) -> dict:
    """The engine log's start-up facts that must not move between arms (expert slots, free VRAM, chunk, workers, switches)."""
    pats = {"engine": r"session is up \(engine ([^)]+)\)", "slots": r"expert cache (\d+) slots",
            "vram_free_mib": r"(\d+) MiB of VRAM free with everything loaded", "chunk": r"prompt chunk auto: (\d+) tokens",
            "workers": r"(\d+) expert-pool workers", "tail_skip": r"(STRATA_ROUTE_TAIL_SKIP=\d+ is ON)",
            "defaults": r"(exact speed switches on by default[^\n]*)", "low_vram_warn": r"(only \d+ MiB of VRAM free[^\n]*)",
            "mem_guard": r"memory guard: limit ([\d.]+ GiB \(source: [^)]*\))", "arena_gib": r"loaded ([\d.]+) GiB at",
            "file_tier_gib": r"file tier ([\d.]+) GiB", "pcie": r"PCIe probe: ([\d.]+) GB/s",
            "engine_cmd": r"engine started: (.*)"}
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return {}
    got = {}
    for k, p in pats.items():
        m = re.findall(p, text)
        if m:
            got[k] = m[-1] if k != "low_vram_warn" else len(m)
    return got


def parse_pairs(spec: str | None, arms: list[str]) -> list[tuple[str, str]] | None:
    if not spec:
        return None
    pairs = []
    for p in spec.split(","):
        name, _, ref = p.strip().partition("/")
        if name not in arms or ref not in arms:
            raise SystemExit(f"--pairs: unknown arm in {p!r} (arms: {', '.join(arms)})")
        pairs.append((ref, name))
    return pairs


def report(out: Path, pairs_spec: str | None = None) -> int:
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text().splitlines() if x.strip()]
    starts = [json.loads(x) for x in (out / "starts.jsonl").read_text().splitlines() if x.strip()]
    arms = list(dict.fromkeys(s["arm"] for s in starts))
    text = summarize(rows, starts, arms, parse_pairs(pairs_spec, arms))
    prof, cmds = [], {}
    for s in starts:
        bp = boot_profile(out / "logs" / f"ab_{s['arm']}_r{s['round']}.log")
        cmd = bp.pop("engine_cmd", None)
        if cmd:
            cmds.setdefault(s["arm"], {}).setdefault(cmd, []).append(s["round"])
        prof.append(f"  r{s['round']} {s['arm']}: " + json.dumps(bp))
    text += "\n\nboot profile per start (must match between rounds of one arm; slots and free VRAM should match between arms):\n" + \
            "\n".join(prof)
    text += "\n\nengine command line per arm, from the server's own 'engine started' line (one entry per distinct line):"
    for arm in arms:
        for cmd, rounds in cmds.get(arm, {}).items():
            text += f"\n  {arm} r{','.join(str(r) for r in rounds)}: {cmd}"
        if arm not in cmds:
            text += f"\n  {arm}: (no 'engine started' line in the engine log)"
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--server", help="serve/server.py of the server version that runs every arm without a server= key")
    ap.add_argument("--body", action="append", default=[], metavar="NAME=JSON",
                    help="fields added to every request body of arm NAME (a JSON object)")
    ap.add_argument("--pairs", help="paired comparisons, e.g. S22/S44,E47pp0/E47 (default: every arm against the first)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true", help="only print the summary of --out")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--order", choices=("reverse", "rotate"), default="reverse")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--probes", default="chat,long")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", type=int, default=6000)
    ap.add_argument("--essay-max", type=int, default=1024)
    ap.add_argument("--sampled-max", type=int, default=2048)
    ap.add_argument("--sampled-think", action="store_true")
    ap.add_argument("--prefill-words", default="21700", help="words of the fresh prompt; a comma list runs one probe per size (pf4k0, ...)")
    ap.add_argument("--prefill-n", type=int, default=2)
    ap.add_argument("--engine", choices=("strata", "mock"), default="strata", help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a.report:
        return report(out, a.pairs)
    if not a.config or len(a.arms) < 1:
        ap.error("CONFIG and at least one ARM are needed")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        arms.append((name, dict(x.split("=", 1) for x in kv.split(",") if x)))
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        ap.error("arm names must differ")
    bodies = {}
    for s in a.body:
        name, _, js = s.partition("=")
        if name not in names:
            ap.error(f"--body {s!r}: no arm {name!r}")
        bodies[name] = json.loads(js)
        if not isinstance(bodies[name], dict):
            ap.error(f"--body {s!r}: not a JSON object")
    for name, env in arms:
        if not env.get("server") and not a.server:
            ap.error(f"arm {name}: no server= key and no --server")
    parse_pairs(a.pairs, names)
    meta = {"argv": sys.argv, "config": base, "bodies": bodies, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "gpu_used_start": gpu_used_mib(), "env": {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    rows_f, starts_f = open(out / "rows.jsonl", "a"), open(out / "starts.jsonl", "a")
    gpu0 = meta["gpu_used_start"]
    for rnd in range(a.rounds):
        if a.order == "reverse":
            order = arms if rnd % 2 == 0 else arms[::-1]
        else:
            k = rnd % len(arms)
            order = arms[k:] + arms[:k]
        for name, env in order:
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(out / "logs" / f"ab_{name}_r{rnd}.log")
            cpath = out / "logs" / f"ab_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            server = Path(e.pop("server", None) or a.server).resolve()
            penv = dict(os.environ, **e)
            cmd = [sys.executable, str(server), "--engine", a.engine, "--port", str(a.port)]
            if a.engine == "strata":
                cmd += ["--config", str(cpath.resolve())]
            st = {"round": rnd, "arm": name, "gpu_before": gpu_used_mib(), "load": loadavg(), "env": e, "server": str(server)}
            srv_log = open(out / "logs" / f"server_{name}_r{rnd}.log", "a")
            proc = subprocess.Popen(cmd, cwd=str(server.parents[1]), env=penv, stdout=srv_log, stderr=subprocess.STDOUT)
            try:
                t0 = time.monotonic()
                wait_ready(a.port, proc)
                st["ready_s"] = round(time.monotonic() - t0)
                print(f"round {rnd} {name}: ready in {st['ready_s']} s (GPU {st['gpu_before']} MiB used before, load {st['load']})",
                      flush=True)
                for tag, body in requests_for(a, rnd, name, bodies.get(name)):
                    r, wall = post(a.port, body)
                    row = row_of(rnd, name, tag, r, wall)
                    rows_f.write(json.dumps(row) + "\n")
                    rows_f.flush()
                    print(f"  {tag:8s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']}, cache {row['cache_n']})"
                          f"  n {row['predicted_n']} drafts {row['draft_ok']}/{row['draft_n']}  {row['sha']}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(90)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(30)
                srv_log.close()
                st["gpu_after"] = wait_gpu_back(gpu0)
                starts_f.write(json.dumps(st) + "\n")
                starts_f.flush()
    rows_f.close()
    starts_f.close()
    return report(out, a.pairs)


if __name__ == "__main__":
    sys.exit(main())
