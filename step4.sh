#!/bin/bash
# pf-q2n on the test machine: build the branch (engine + the fused prompt kernel test), the kernel test, an engine A/B of the
# prompt speed, and first-token KL of long prompts between the paths.
#   scp payload/src-pf-q2n.tar.gz <gpu-host>:strata-lab/ab/src/
#   scp payload/build_engines.sh payload/ab_probe.py payload/kl_prompt.py <gpu-host>:strata-lab/ab/tools/
#   ssh <gpu-host> 'bash -s' < step4.sh
# Arms of the A/B (serve/ of v0.2.2 for all; STRATA_PREFILL_CPU_SHARE=0 so the prompt pass is the GPU's alone):
#   stock  engine 0.1.47 as released (v0.2.2 tree), variable unset
#   base   the branch, variable unset            base/stock  = the change does nothing unset
#   q2n    the branch, STRATA_PF_FUSED_Q2N=1     q2n/base    = the fused int8 kernels for the Q2_0 gate/up
set -u
R=$HOME/strata-lab
AB=$R/ab
ROUNDS=${ROUNDS:-5}
RUN=step4-$(date +%m%d-%H%M)
OUT=$AB/results/$RUN
cd "$AB" || exit 2
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 8192 ] || { echo "only $free MiB of VRAM free (8192 needed): stopping"; exit 4; }
if docker ps --format '{{.Names}}' | grep -qE '^strata-lab'; then echo "another container of ours is running: stopping"; exit 4; fi
for f in src/src-pf-q2n.tar.gz tools/ab_probe.py tools/kl_prompt.py tools/build_engines.sh engines/v0.2.2/strata cfg/ab-q2_0-static.json; do
  [ -e "$f" ] || { echo "missing $f"; exit 2; }
done
grep -q 'TARGETS=' tools/build_engines.sh || { echo "tools/build_engines.sh is the old one"; exit 2; }
grep -q 'prefill-words' tools/ab_probe.py && grep -q 'pf{words' tools/ab_probe.py || { echo "tools/ab_probe.py is the old one"; exit 2; }
mkdir -p "$OUT/tmp" cfg home state config
nvidia-smi > "$OUT/nvidia-smi-before.txt"; cat /proc/loadavg > "$OUT/loadavg-before.txt"
# 1. build the branch: the engine and the kernel test
if [ ! -x engines/pf-q2n/strata ] || [ ! -x engines/pf-q2n/prefill_fused_iq_test ]; then
  rm -rf src/src-pf-q2n; tar xzf src/src-pf-q2n.tar.gz -C src || exit 2
  docker rm -f strata-lab-ab-build > /dev/null 2>&1
  t1=$(date +%s)
  docker run --name strata-lab-ab-build --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
    -e HOME=/ab/builder -e MAX_JOBS=8 -e TESTS=ON -e TARGETS="strata prefill_fused_iq_test" -v "$AB:/ab" \
    strata-lab-ab-builder sh /ab/tools/build_engines.sh pf-q2n 2>&1 | tail -4
  echo "== build done in $(( $(date +%s) - t1 )) s, container rc=$(docker inspect -f '{{.State.ExitCode}}' strata-lab-ab-build)"
  docker rm strata-lab-ab-build > /dev/null 2>&1
  [ -x engines/pf-q2n/strata ] && [ -x engines/pf-q2n/prefill_fused_iq_test ] || { echo "build failed"; tail -20 engines/pf-q2n/build-strata.log; exit 3; }
  cat engines/pf-q2n/build-time.json
fi
common=(--gpus device=0 --cpuset-cpus 0-15 --memory 96g --memory-swap 96g --ulimit memlock=-1 --shm-size 2g --network none
        --user "$(id -u):$(id -g)" -v "$R/data:/data:ro" -v "$AB:/ab"
        -e HOME=/ab/home -e XDG_STATE_HOME=/ab/state -e XDG_CONFIG_HOME=/ab/config -e TMPDIR="/ab/results/$RUN/tmp"
        -e PYTHONNOUSERSITE=1 -e STRATA_PREFILL_CPU_SHARE=0 -e LD_LIBRARY_PATH=/ab/cudalibs)
# 2. the kernel test on the Q2_0 / Q2_0 pair: parity against the FP32 reference and MMQ, then timing at the engine's chunks
echo "== $(date +%T) kernel test"
docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" -e STRATA_PF_FUSED_Q2N=1 strata-lab-base \
  /ab/engines/pf-q2n/prefill_fused_iq_test "--only=Q2_0 / Q2_0" --chunks=2048,4864,8192 > "$OUT/kernel-test.log" 2>&1
echo "kernel test rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1
grep -E 'rel RMS|ms|passed|failed|skipped' "$OUT/kernel-test.log" | head -30
# 3. engine A/B: prompt speed on fresh 4K / 16K / 23K prompts (2 each per start) and the chat probes, 5 rounds rotating
echo "== $(date +%T) engine A/B"
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/ab_probe.py \
    /ab/cfg/ab-q2_0-static.json \
    stock:exe=/ab/engines/v0.2.2/strata base:exe=/ab/engines/pf-q2n/strata "q2n:exe=/ab/engines/pf-q2n/strata,STRATA_PF_FUSED_Q2N=1" \
    --server /ab/src/src-v0.2.2/serve/server.py --pairs base/stock,q2n/base \
    --out "/ab/results/$RUN/ab" --rounds "$ROUNDS" --order rotate --probes chat,prefill --prefill-words 4000,15000,21700 --prefill-n 2 \
    > "$OUT/ab.console.log" 2>&1
echo "A/B rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1
head -12 "$OUT/ab/summary.txt"
# 4. first-token KL of 8 long prompts: the stock engine, the branch unset (must be byte-identical), the branch with the variable,
#    and the FP16 prompt path (STRATA_PREFILL_MMQ=0) as the reference both int8 paths are measured against
echo "== $(date +%T) KL"
IDS=/ab/src/src-pf-q2n/data/quant_quality/v1/set.ids.json
for arm in stock:v0.2.2: base:pf-q2n: q2n:pf-q2n:STRATA_PF_FUSED_Q2N=1 fp16:pf-q2n:STRATA_PREFILL_MMQ=0; do
  name=${arm%%:*}; rest=${arm#*:}; eng=${rest%%:*}; envkv=${rest#*:}
  extra=(); [ -n "$envkv" ] && extra=(--env "$envkv")
  docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py run \
      /ab/cfg/ab-q2_0-static.json --exe "/ab/engines/$eng/strata" --out "/ab/results/$RUN/kl/$name" --ids "$IDS" "${extra[@]}" \
      > "$OUT/kl-$name.console.log" 2>&1
  echo "kl $name rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1
  tail -1 "$OUT/kl-$name.console.log"
done
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/stock" "/ab/results/$RUN/kl/base" > "$OUT/kl-identity.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/fp16" "/ab/results/$RUN/kl/base" "/ab/results/$RUN/kl/q2n" > "$OUT/kl-vs-fp16.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/base" "/ab/results/$RUN/kl/q2n" > "$OUT/kl-vs-mmq.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
head -3 "$OUT/kl-identity.txt"; head -2 "$OUT/kl-vs-fp16.txt"; sed -n 1p "$OUT/kl-vs-mmq.txt"
nvidia-smi > "$OUT/nvidia-smi-after.txt"
echo "== $(date +%T) $RUN done"
