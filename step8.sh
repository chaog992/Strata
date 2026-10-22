#!/bin/bash
# pf-q2n step 8 on the test machine: the ctest entry now runs with STRATA_PF_FUSED empty (the opt-in alone), plus two manual
# checks of the switches: STRATA_PF_FUSED=0 turns the opt-in off (the test then refuses the pair: exit 1), and the opt-in
# alone also turns the engine's fused path on (the engine log line). Rebuilds the final commit from src-pf-q2n3.tar.gz.
#   scp payload/src-pf-q2n3.tar.gz <gpu-host>:strata-lab/ab/src/
#   ssh <gpu-host> 'cd ~/strata-lab/ab && nohup setsid bash tools/step8.sh > results/step8.console.log 2>&1 < /dev/null &'
set -u
R=$HOME/strata-lab
AB=$R/ab
RUN=step8-$(date +%m%d-%H%M)
OUT=$AB/results/$RUN
cd "$AB" || exit 2
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 8192 ] || { echo "only $free MiB of VRAM free (8192 needed): stopping"; exit 4; }
if docker ps --format '{{.Names}}' | grep -qE '^strata-lab'; then echo "another container of ours is running: stopping"; exit 4; fi
mkdir -p "$OUT/tmp"
rm -rf src/src-pf-q2n3 engines/pf-q2n3 build/pf-q2n3; tar xzf src/src-pf-q2n3.tar.gz -C src || exit 2
docker rm -f strata-lab-ab-build > /dev/null 2>&1
t1=$(date +%s)
docker run --name strata-lab-ab-build --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -e MAX_JOBS=8 -e TESTS=ON -e TARGETS="strata prefill_fused_iq_test" -v "$AB:/ab" \
  strata-lab-ab-builder sh /ab/tools/build_engines.sh pf-q2n3 2>&1 | tail -2
echo "== build done in $(( $(date +%s) - t1 )) s, rc=$(docker inspect -f '{{.State.ExitCode}}' strata-lab-ab-build)"
docker rm strata-lab-ab-build > /dev/null 2>&1
[ -x engines/pf-q2n3/strata ] && [ -x engines/pf-q2n3/prefill_fused_iq_test ] || { echo "build failed"; exit 3; }
sha256sum engines/pf-q2n3/strata engines/pf-q2n2/strata | cut -c1-16,64-
echo "== $(date +%T) ctest (the q2n entry with STRATA_PF_FUSED empty)"
docker run --name strata-lab-ab-build --gpus device=0 --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -v "$AB:/ab" strata-lab-ab-builder sh -c 'cd /ab/build/pf-q2n3 && ctest -R prefill_fused_iq_test -V 2>&1 | grep -E "Environment|STRATA_PF|Test #|tests passed|prompt experts|skipped|Passed|Failed" | head -20' > "$OUT/ctest.log" 2>&1
echo "ctest rc=$?"; docker rm strata-lab-ab-build > /dev/null 2>&1; cat "$OUT/ctest.log"
common=(--gpus device=0 --cpuset-cpus 0-15 --memory 96g --memory-swap 96g --ulimit memlock=-1 --shm-size 2g --network none
        --user "$(id -u):$(id -g)" -v "$R/data:/data:ro" -v "$AB:/ab"
        -e HOME=/ab/home -e XDG_STATE_HOME=/ab/state -e XDG_CONFIG_HOME=/ab/config -e TMPDIR="/ab/results/$RUN/tmp"
        -e PYTHONNOUSERSITE=1 -e STRATA_PREFILL_CPU_SHARE=0 -e LD_LIBRARY_PATH=/ab/cudalibs)
T=/ab/engines/pf-q2n3/prefill_fused_iq_test
echo "== $(date +%T) switch checks with the test binary (--only insists, so 'not covered' = the path is off)"
for c in "Q2N alone:STRATA_PF_FUSED_Q2N=1" "Q2N + PF_FUSED=0:STRATA_PF_FUSED_Q2N=1,STRATA_PF_FUSED=0" "Q2N + PF_FUSED=1:STRATA_PF_FUSED_Q2N=1,STRATA_PF_FUSED=1" "nothing set:NONE"; do
  name=${c%%:*}; envs=${c#*:}; args=()
  [ "$envs" != NONE ] && for kv in ${envs//,/ }; do args+=(-e "$kv"); done
  docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
  docker run --name strata-lab-ab-q2n "${common[@]}" "${args[@]}" strata-lab-base $T "--only=Q2_0 / Q2_0" --no-timing > "$OUT/switch.log" 2>&1
  echo "  $name: rc=$? :: $(grep -E 'not covered|passed|skipped' "$OUT/switch.log" | tail -1)"
  docker rm strata-lab-ab-q2n > /dev/null 2>&1
done
echo "== $(date +%T) the engine with the opt-in alone: the fused line, 1 prompt"
IDS=/ab/src/src-pf-q2n3/data/quant_quality/v1/set.ids.json
docker run --name strata-lab-ab-q2n "${common[@]}" -e STRATA_PF_FUSED_Q2N=1 strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py run \
    /ab/cfg/ab-q2_0-static.json --exe /ab/engines/pf-q2n3/strata --out "/ab/results/$RUN/kl/alone" --ids "$IDS" --lens 2048 > "$OUT/kl-alone.console.log" 2>&1
echo "engine rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1; tail -1 "$OUT/kl-alone.console.log"
nvidia-smi > "$OUT/nvidia-smi-after.txt"
echo "== $(date +%T) $RUN done"
