#!/bin/bash
# pf-q2n step 7 on the test machine: the FINAL commit's engine and test - build, the new ctest entry, the kernel test with the
# variable, and the unset identity check (stock v0.2.2 engine vs the final engine, 40 fresh prompts, prompt cache off).
#   scp payload/src-pf-q2n2.tar.gz <gpu-host>:strata-lab/ab/src/
#   ssh <gpu-host> 'cd ~/strata-lab/ab && nohup setsid bash tools/step7.sh > results/step7.console.log 2>&1 < /dev/null &'
set -u
R=$HOME/strata-lab
AB=$R/ab
RUN=step7-$(date +%m%d-%H%M)
OUT=$AB/results/$RUN
cd "$AB" || exit 2
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 8192 ] || { echo "only $free MiB of VRAM free (8192 needed): stopping"; exit 4; }
if docker ps --format '{{.Names}}' | grep -qE '^strata-lab'; then echo "another container of ours is running: stopping"; exit 4; fi
for f in src/src-pf-q2n2.tar.gz tools/kl_prompt.py tools/build_engines.sh engines/v0.2.2/strata cfg/ab-q2_0-static.json; do [ -e "$f" ] || { echo "missing $f"; exit 2; }; done
mkdir -p "$OUT/tmp"
nvidia-smi > "$OUT/nvidia-smi-before.txt"; cat /proc/loadavg > "$OUT/loadavg-before.txt"
# 1. build the final commit (engine + test), TESTS=ON so the ctest entries exist in the build tree
rm -rf src/src-pf-q2n2 engines/pf-q2n2 build/pf-q2n2; tar xzf src/src-pf-q2n2.tar.gz -C src || exit 2
docker rm -f strata-lab-ab-build > /dev/null 2>&1
t1=$(date +%s)
docker run --name strata-lab-ab-build --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -e MAX_JOBS=8 -e TESTS=ON -e TARGETS="strata prefill_fused_iq_test" -v "$AB:/ab" \
  strata-lab-ab-builder sh /ab/tools/build_engines.sh pf-q2n2 2>&1 | tail -3
echo "== build done in $(( $(date +%s) - t1 )) s, rc=$(docker inspect -f '{{.State.ExitCode}}' strata-lab-ab-build)"
docker rm strata-lab-ab-build > /dev/null 2>&1
[ -x engines/pf-q2n2/strata ] && [ -x engines/pf-q2n2/prefill_fused_iq_test ] || { echo "build failed"; exit 3; }
sha256sum engines/pf-q2n2/strata engines/pf-q2n/strata | cut -c1-16,64-
# 2. the ctest entries, in the builder image with the card (ctest lives there)
echo "== $(date +%T) ctest"
docker run --name strata-lab-ab-build --gpus device=0 --cpuset-cpus 0-15 --memory 48g --memory-swap 48g --user "$(id -u):$(id -g)" \
  -e HOME=/ab/builder -v "$AB:/ab" strata-lab-ab-builder sh -c 'cd /ab/build/pf-q2n2 && ctest -R prefill_fused_iq_test --output-on-failure 2>&1 | tail -8' > "$OUT/ctest.log" 2>&1
echo "ctest rc=$?"; docker rm strata-lab-ab-build > /dev/null 2>&1; cat "$OUT/ctest.log"
common=(--gpus device=0 --cpuset-cpus 0-15 --memory 96g --memory-swap 96g --ulimit memlock=-1 --shm-size 2g --network none
        --user "$(id -u):$(id -g)" -v "$R/data:/data:ro" -v "$AB:/ab"
        -e HOME=/ab/home -e XDG_STATE_HOME=/ab/state -e XDG_CONFIG_HOME=/ab/config -e TMPDIR="/ab/results/$RUN/tmp"
        -e PYTHONNOUSERSITE=1 -e STRATA_PREFILL_CPU_SHARE=0 -e LD_LIBRARY_PATH=/ab/cudalibs)
# 3. the kernel test with the variable, at the engine's chunks (the same command as step 4)
echo "== $(date +%T) kernel test"
docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" -e STRATA_PF_FUSED_Q2N=1 strata-lab-base \
  /ab/engines/pf-q2n2/prefill_fused_iq_test "--only=Q2_0 / Q2_0" --chunks=2048,4864,8192 > "$OUT/kernel-test.log" 2>&1
echo "kernel test rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1; grep -E 'timing part|passed|failed' "$OUT/kernel-test.log"
# 4. identity: stock v0.2.2 engine vs the final engine, both unset, 40 fresh prompts, prompt cache off; plus the final engine set
IDS=/ab/src/src-pf-q2n2/data/quant_quality/v1/set.ids.json
for arm in stock:v0.2.2: final:pf-q2n2: finalq2n:pf-q2n2:STRATA_PF_FUSED_Q2N=1; do
  name=${arm%%:*}; rest=${arm#*:}; eng=${rest%%:*}; envkv=${rest#*:}
  extra=(); [ -n "$envkv" ] && extra=(--env "$envkv")
  docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py run \
      /ab/cfg/ab-q2_0-static.json --exe "/ab/engines/$eng/strata" --out "/ab/results/$RUN/kl/$name" --ids "$IDS" --fresh 40 --xargs "--prompt-cache 0" "${extra[@]}" \
      > "$OUT/kl-$name.console.log" 2>&1
  echo "kl $name rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1; tail -1 "$OUT/kl-$name.console.log"
done
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/stock" "/ab/results/$RUN/kl/final" > "$OUT/kl-identity.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/final" "/ab/results/$RUN/kl/finalq2n" > "$OUT/kl-final-vs-q2n.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
head -1 "$OUT/kl-identity.txt"; head -1 "$OUT/kl-final-vs-q2n.txt"
nvidia-smi > "$OUT/nvidia-smi-after.txt"
echo "== $(date +%T) $RUN done"
