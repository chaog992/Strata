"""Make the A/B base config from a run config that `strata launch` wrote (/data/install/strata-<tag>.json).

    python3 make_ab_config.py IN.json OUT.json [--lib-dir /ab/cudalibs] [--args-line "<engine args>"]

What it changes, and why (the arms must differ only in the engine):
  * host 127.0.0.1, no api_key (tools/ab_engine.py sends no Authorization header)
  * lib_dirs = [--lib-dir]: one CUDA runtime for every arm (the server puts lib_dirs first on LD_LIBRARY_PATH)
  * drops parallel/replicas/expert_profile_save*/vision/idle_unload: one request at a time, no learned profile carried
    from one arm to the next (serve/server.py learned_profile_args), no picture reader
  * --args-line replaces "args" (e.g. the TUNE=skip line of an earlier engine log); "user_args" and "env" are printed,
    because the v0.2.2 server merges them at every start (engine_args, engine_env)
It refuses engine flags that engine 0.1.44.1 does not have (all four are opt-in in v0.2.2).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys

NEW_SINCE_0441 = ("--conversation-cache-park-gain", "--pool-cores", "--prefill-pipe", "--prompt-cache-spread")
DROP = ("api_key", "parallel", "replicas", "expert_profile_save", "expert_profile_save_every", "vision", "idle_unload",
        "open_browser", "mcp", "prevent_sleep", "pool_cores", "vram_elastic", "split_skip_if_fits")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--lib-dir", default="/ab/cudalibs")
    ap.add_argument("--args-line", default=None, help="engine arguments as one shell-quoted line (replaces \"args\")")
    ap.add_argument("--drop-env", action="store_true", help="drop \"env\" (e.g. a tuning run's engine settings)")
    ap.add_argument("--drop-user-args", action="store_true", help="drop \"user_args\" (then --args-line is the whole line)")
    a = ap.parse_args()
    cfg = json.load(open(a.src, encoding="utf-8-sig"))
    out = dict(cfg)
    for k in DROP + (("env",) if a.drop_env else ()) + (("user_args", "args_written") if a.drop_user_args else ()):
        out.pop(k, None)
    out["host"] = "127.0.0.1"
    out["lib_dirs"] = [a.lib_dir]
    if a.args_line:
        out["args"] = shlex.split(a.args_line)
    args = [x for x in out["args"] if x != "--vision"]
    out["args"] = args
    bad = [x for x in args + list(out.get("user_args") or []) if isinstance(x, str) and x.split("=")[0] in NEW_SINCE_0441]
    print("exe        :", cfg.get("exe"))
    print("args       :", " ".join(args))
    print("user_args  :", out.get("user_args"))
    print("env        :", out.get("env"))
    print("cwd        :", out.get("cwd"), "| tokenizer:", out.get("tokenizer"), "| arch:", out.get("arch"))
    print("dropped    :", [k for k in DROP if k in cfg])
    if bad:
        print("REFUSED: flags engine 0.1.44.1 does not know:", bad, file=sys.stderr)
        return 3
    json.dump(out, open(a.dst, "w", encoding="utf-8"), indent=1)
    print("wrote", a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
