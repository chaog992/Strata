"""First-token logits of long prompts under one engine arm, and the KL between arms (scratch tool, not in the repo).

    python3 kl_prompt.py run CONFIG --exe EXE --out DIR [--env K=V ...] [--ids set.ids.json] [--lens 1536,2048,3072,5000]
    python3 kl_prompt.py compare REF_DIR RUN_DIR [RUN_DIR ...]

`run` starts `EXE --serve` with CONFIG's args (an ab_probe / make_ab_config config: args, env, lib_dirs, cwd), sends one
`GEN 1 <ids>` per prompt and lets STRATA_DUMP_FIRST_LOGITS write the logits after each prompt (the whole vocabulary, float32),
the way tools/quant_quality.py's `prompt` mode does - but with prompts of 1.5K-5K tokens, above the 1024-token chunk from
which every expert streams and the fused prompt kernels (moe_fused_iq) can take a layer.  The prompts are the quant-quality
set's five 1,000-token segments (data/quant_quality/v1/set.ids.json) concatenated in two orders and cut to --lens, so 8
prompts by default.  The engine's stderr goes to DIR/stderr.txt (the "prompt experts on the fused int8 kernels" line says
which path ran).

`compare` prints, per prompt and overall: KL(REF || RUN) over the full vocabulary (natural log), top-1 agreement, the
largest absolute logit difference, and whether the two files are byte-identical.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

SEG_ORDERS = ((0, 1, 2, 3, 4), (4, 2, 0, 3, 1))
# --fresh: eight prompts that share no prefix (a different first segment or a different second one), one length each,
# so with --xargs "--prompt-cache 0" every prompt is one fresh read of >= 1024 tokens (the fused path's chunk floor)
FRESH = (((0, 1, 2, 3, 4), 1536), ((1, 2, 3, 4, 0), 2048), ((2, 3, 4, 0, 1), 3072), ((3, 4, 0, 1, 2), 4096),
         ((4, 0, 1, 2, 3), 5000), ((0, 3, 1, 4, 2), 2560), ((1, 4, 2, 0, 3), 3584), ((2, 0, 4, 1, 3), 4608))


def prompts(ids_file: str, lens: list[int], fresh: int = 0) -> list[tuple[str, list[int]]]:
    st = json.load(open(ids_file))
    segs = [s["ids"] for s in st["segments"]]
    out = []
    if fresh:
        # --fresh 8: FRESH; --fresh N > 8: N prompts over the 120 segment orders, lengths cycling 1024..5000 (one chunk each)
        import itertools
        plan = list(FRESH)
        if fresh > len(FRESH):
            lens_cycle = (1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5000)
            orders = [o for o in itertools.permutations(range(5)) if (o, None) not in FRESH]
            rng_orders = orders[::max(1, len(orders) // (fresh - len(FRESH)))]
            plan += [(o, lens_cycle[i % len(lens_cycle)]) for i, o in enumerate(rng_orders)]
        for order, n in plan[:fresh]:
            cat = [t for i in order for t in segs[i]]
            out.append((f"f{''.join(map(str, order))}-{n}", cat[:n]))
        return out
    for oi, order in enumerate(SEG_ORDERS):
        cat = [t for i in order for t in segs[i]]
        for n in lens:
            if n <= len(cat):
                out.append((f"o{oi}-{n}", cat[:n]))
    return out


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    out = pathlib.Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("first.*"):
        f.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
    env["STRATA_PREFILL_CPU_SHARE"] = "0"          # the prompt pass on the GPU only (the share is wall-clock dependent)
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    env["STRATA_DUMP_FIRST_LOGITS"] = str(out / "first")
    args = [a.exe, "--serve", *cfg["args"], *(a.xargs.split() if a.xargs else [])]
    ps = prompts(a.ids, [int(x) for x in a.lens.split(",")], a.fresh)
    errf = open(out / "stderr.txt", "w")
    print("engine:", " ".join(args), flush=True)
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True,
                         cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"the engine ended before READY (exit {p.wait()}); see {out}/stderr.txt")
    index = []
    for name, ids in ps:
        p.stdin.write("GEN 1 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        done = None
        for line in p.stdout:
            if line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
        if done is None or not done.startswith("DONE"):
            p.kill()
            raise SystemExit(f"{name}: the engine answered {done!r}; see {out}/stderr.txt")
        index.append({"name": name, "tokens": len(ids), "done": done[:200]})   # the DONE line has chunk= and experts_streamed=
        print(f"  {name}: {len(ids)} tokens, {done[:120]}", flush=True)
    p.stdin.write("QUIT\n")
    p.stdin.flush()
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    err = (out / "stderr.txt").read_text(errors="replace")
    fused = [l for l in err.splitlines() if "fused int8" in l]
    (out / "index.json").write_text(json.dumps({"requests": index, "args": args, "env": a.env or [], "fused_lines": fused},
                                               indent=1))
    print(f"{len(index)} prompts; fused-path lines in stderr: {len(fused)}")
    return 0


def cmd_compare(a) -> int:
    import numpy as np
    ref = pathlib.Path(a.ref)
    idx = json.loads((ref / "index.json").read_text())["requests"]

    def logp(d: pathlib.Path, i: int):
        x = np.fromfile(d / f"first.{i}", dtype=np.float32).astype(np.float64)
        x -= x.max()
        return x - np.log(np.exp(x).sum())

    for run in a.runs:
        rd = pathlib.Path(run)
        rows, kls, tops, same_all = [], [], [], True
        for i, r in enumerate(idx):
            pr, qr = logp(ref, i), logp(rd, i)
            kl = float(np.sum(np.exp(pr) * (pr - qr)))
            top = int(np.argmax(pr) == np.argmax(qr))
            raw_a = np.fromfile(ref / f"first.{i}", dtype=np.float32)
            raw_b = np.fromfile(rd / f"first.{i}", dtype=np.float32)
            same = bool(np.array_equal(raw_a, raw_b))
            same_all &= same
            kls.append(kl)
            tops.append(top)
            rows.append(f"    {r['name']:9s} {r['tokens']:5d} tok  KL {kl:.5f}  top-1 {'same' if top else 'DIFF'}  "
                        f"max|dlogit| {float(np.max(np.abs(raw_a - raw_b))):.4f}  {'identical' if same else ''}")
        print(f"{rd.name} vs {ref.name}: mean KL {np.mean(kls):.5f}, max {np.max(kls):.5f}, top-1 agreement "
              f"{sum(tops)}/{len(tops)}, {'ALL FILES BYTE-IDENTICAL' if same_all else 'files differ'}")
        print("\n".join(rows))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--exe", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", action="append")
    p.add_argument("--ids", default="data/quant_quality/v1/set.ids.json")
    p.add_argument("--lens", default="1536,2048,3072,5000")
    p.add_argument("--fresh", type=int, nargs="?", const=8, default=0, help="N fresh prompts (8: FRESH; more: over the segment orders)")
    p.add_argument("--xargs", help="engine arguments appended to the config's, e.g. '--prompt-cache 0'")
    p = sub.add_parser("compare")
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    a = ap.parse_args()
    return {"run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
