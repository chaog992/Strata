#!/bin/bash
# pf-q2n step 9 on the lab box: the final commit (test changes only against step 8's build: --unset-pf-fused, the skip below
# sm_80 with STRATA_PF_FUSED unset, the ctest entry without an empty STRATA_PF_FUSED; src/ unchanged). Rebuilds engine + test
# from src-pf-q2n4.tar.gz, then: ctest (both entries), the switch checks with the test binary (the flag, an emulated Turing
# card for the skip, and step 8's test binary under the same emulation for the failure it fixes), the engine with the opt-in
# alone and with STRATA_PF_FUSED=0 added (the fused line present / absent), and the kernel test with the flag (timing).
#   scp payload/src-pf-q2n4.tar.gz <gpu-host>:strata-lab/ab/src/
#   ssh <gpu-host> 'cd ~/strata-lab/ab && nohup setsid bash tools/step9.sh > results/step9.console.log 2>&1 < /dev/null &'
set -u
R=$HOME/strata-lab
AB=$R/ab
RUN=step9-$(date +%m%d-%H%M)
OUT=$AB/results/$RUN
cd "$AB" || exit 2
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 8192 ] || { echo "only $free MiB of VRAM free (8192 needed): stopping"; exit 4; }
if docker ps --format '{{.Names}}' | grep -qE '^strata-lab'; then echo "another container of ours is running: stopping"; exit 4; fi
[ -f src/src-pf-q2n4.tar.gz ] || { echo "missing src/src-pf-q2n4.tar.gz"; exit 2; }
[ -x engines/pf-q2n3/prefill_fused_iq_test ] || { echo "missing step 8's test binary (engines/pf-q2n3)"; exit 2; }
mkdir -p "$OUT/tmp"
nvidia-smi > "$OUT/nvidia-smi-before.txt"; cat /proc/loadavg > "$OUT/loadavg-before.txt"
rm -rf src/src-pf-q2n4 engines/pf-q2n4 build/pf-q2n4; tar xzf src/src-pf-q2n4.tar.gz -C src || exit 2
docker rm -f strata-lab-ab-build > /dev/null 2>&1
t1=$(date +%s)
docker run --name strata-lab-ab-build --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -e MAX_JOBS=8 -e TESTS=ON -e TARGETS="strata prefill_fused_iq_test" -v "$AB:/ab" \
  strata-lab-ab-builder sh /ab/tools/build_engines.sh pf-q2n4 2>&1 | tail -2
echo "== build done in $(( $(date +%s) - t1 )) s, rc=$(docker inspect -f '{{.State.ExitCode}}' strata-lab-ab-build)"
docker rm strata-lab-ab-build > /dev/null 2>&1
[ -x engines/pf-q2n4/strata ] && [ -x engines/pf-q2n4/prefill_fused_iq_test ] || { echo "build failed"; exit 3; }
sha256sum engines/pf-q2n4/strata engines/pf-q2n3/strata engines/pf-q2n4/prefill_fused_iq_test engines/pf-q2n3/prefill_fused_iq_test | cut -c1-16,64- | tee "$OUT/engine-sha256.txt"
echo "== $(date +%T) ctest (both entries; the q2n entry with --unset-pf-fused and STRATA_PF_FUSED_Q2N=1 only)"
docker run --name strata-lab-ab-build --gpus device=0 --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -v "$AB:/ab" strata-lab-ab-builder sh -c 'cd /ab/build/pf-q2n4 && ctest -R prefill_fused_iq_test -V 2>&1 | grep -E "Environment|STRATA_PF|Test command|Test #|tests passed|prompt experts|skipped|Passed|Failed" | head -24' > "$OUT/ctest.log" 2>&1
echo "ctest rc=$?"; docker rm strata-lab-ab-build > /dev/null 2>&1; cat "$OUT/ctest.log"
common=(--gpus device=0 --cpuset-cpus 0-15 --memory 96g --memory-swap 96g --ulimit memlock=-1 --shm-size 2g --network none
        --user "$(id -u):$(id -g)" -v "$R/data:/data:ro" -v "$AB:/ab"
        -e HOME=/ab/home -e XDG_STATE_HOME=/ab/state -e XDG_CONFIG_HOME=/ab/config -e TMPDIR="/ab/results/$RUN/tmp"
        -e PYTHONNOUSERSITE=1 -e STRATA_PREFILL_CPU_SHARE=0 -e LD_LIBRARY_PATH=/ab/cudalibs)
T=/ab/engines/pf-q2n4/prefill_fused_iq_test
T3=/ab/engines/pf-q2n3/prefill_fused_iq_test
echo "== $(date +%T) switch checks with the test binary (--only='Q2_0 / Q2_0' --no-timing; --only insists, so 'not covered' = the pair is off)"
i=0
# name : environment (NONE = nothing) : extra test arguments : binary
for c in \
  "flag, Q2N alone (the ctest entry):STRATA_PF_FUSED_Q2N=1:--unset-pf-fused:$T" \
  "flag, nothing set:NONE:--unset-pf-fused:$T" \
  "no flag, Q2N (the test sets PF_FUSED=1):STRATA_PF_FUSED_Q2N=1::$T" \
  "no flag, nothing set (PF_FUSED=1 from the test):NONE::$T" \
  "flag, Q2N, emulated 7.5 card with SM75=1 (this commit's test):STRATA_PF_FUSED_Q2N=1,STRATA_EMULATE_CC=75,STRATA_PF_FUSED_SM75=1:--unset-pf-fused:$T" \
  "step 8's test under the same emulation with PF_FUSED empty (step 8's ctest entry):STRATA_PF_FUSED_Q2N=1,STRATA_EMULATE_CC=75,STRATA_PF_FUSED_SM75=1,STRATA_PF_FUSED=::$T3" \
  "flag, Q2N, emulated 8.6 card (this commit's test):STRATA_PF_FUSED_Q2N=1,STRATA_EMULATE_CC=86:--unset-pf-fused:$T"; do
  IFS=: read -r name envs extra bin <<< "$c"; args=(); xargs=()
  [ "$envs" != NONE ] && for kv in ${envs//,/ }; do args+=(-e "$kv"); done
  [ -n "$extra" ] && xargs=("$extra")
  i=$((i + 1))
  docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
  docker run --name strata-lab-ab-q2n "${common[@]}" "${args[@]}" strata-lab-base "$bin" "--only=Q2_0 / Q2_0" --no-timing "${xargs[@]}" > "$OUT/switch-$i.log" 2>&1
  echo "  $name: rc=$? :: $(grep -E 'not covered|passed|skipped' "$OUT/switch-$i.log" | tail -1)" | tee -a "$OUT/switch-checks.txt"
  docker rm strata-lab-ab-q2n > /dev/null 2>&1
done
echo "== $(date +%T) the engine with the opt-in alone, and with STRATA_PF_FUSED=0 added (the fused line present / absent), 2 prompts each"
IDS=/ab/src/src-pf-q2n4/data/quant_quality/v1/set.ids.json
for arm in alone:STRATA_PF_FUSED_Q2N=1 off:STRATA_PF_FUSED_Q2N=1,STRATA_PF_FUSED=0; do
  name=${arm%%:*}; envs=${arm#*:}; args=()
  for kv in ${envs//,/ }; do args+=(-e "$kv"); done
  docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
  docker run --name strata-lab-ab-q2n "${common[@]}" "${args[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py run \
      /ab/cfg/ab-q2_0-static.json --exe /ab/engines/pf-q2n4/strata --out "/ab/results/$RUN/kl/$name" --ids "$IDS" --lens 2048 > "$OUT/kl-$name.console.log" 2>&1
  echo "engine $name ($envs) rc=$? :: $(tail -1 "$OUT/kl-$name.console.log")"; docker rm strata-lab-ab-q2n > /dev/null 2>&1
done
echo "== $(date +%T) the kernel test with the flag (timing at 2048 / 4864 / 8192)"
docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" -e STRATA_PF_FUSED_Q2N=1 strata-lab-base "$T" "--only=Q2_0 / Q2_0" --unset-pf-fused --chunks=2048,4864,8192 > "$OUT/kernel-test.log" 2>&1
echo "kernel test rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1; grep -E "ms|passed|rel RMS|fused|MMQ" "$OUT/kernel-test.log" | head -20
nvidia-smi > "$OUT/nvidia-smi-after.txt"
echo "== $(date +%T) $RUN done"
