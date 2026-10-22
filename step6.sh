#!/bin/bash
# pf-q2n step 6 on the test machine: first-token KL on 40 fresh single-chunk prompts (prompt cache off), three arms:
# MMQ (unset), fused (STRATA_PF_FUSED_Q2N=1) and the FP16 prompt path (STRATA_PREFILL_MMQ=0) as the reference.
#   scp payload/kl_prompt.py <gpu-host>:strata-lab/ab/tools/
#   ssh <gpu-host> 'bash -s' < step6.sh
set -u
R=$HOME/strata-lab
AB=$R/ab
N=${N:-40}
RUN=step6-$(date +%m%d-%H%M)
OUT=$AB/results/$RUN
cd "$AB" || exit 2
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "$free" -ge 8192 ] || { echo "only $free MiB of VRAM free (8192 needed): stopping"; exit 4; }
if docker ps --format '{{.Names}}' | grep -qE '^strata-lab'; then echo "another container of ours is running: stopping"; exit 4; fi
for f in tools/kl_prompt.py engines/pf-q2n/strata cfg/ab-q2_0-static.json; do [ -e "$f" ] || { echo "missing $f"; exit 2; }; done
grep -q 'lens_cycle' tools/kl_prompt.py || { echo "tools/kl_prompt.py is the old one"; exit 2; }
mkdir -p "$OUT/tmp"
nvidia-smi > "$OUT/nvidia-smi-before.txt"; cat /proc/loadavg > "$OUT/loadavg-before.txt"
common=(--gpus device=0 --cpuset-cpus 0-15 --memory 96g --memory-swap 96g --ulimit memlock=-1 --shm-size 2g --network none
        --user "$(id -u):$(id -g)" -v "$R/data:/data:ro" -v "$AB:/ab"
        -e HOME=/ab/home -e XDG_STATE_HOME=/ab/state -e XDG_CONFIG_HOME=/ab/config -e TMPDIR="/ab/results/$RUN/tmp"
        -e PYTHONNOUSERSITE=1 -e STRATA_PREFILL_CPU_SHARE=0 -e LD_LIBRARY_PATH=/ab/cudalibs)
E=/ab/engines/pf-q2n/strata
IDS=/ab/src/src-pf-q2n/data/quant_quality/v1/set.ids.json
echo "== $(date +%T) KL on $N fresh prompts"
docker rm -f strata-lab-ab-q2n > /dev/null 2>&1
for arm in base: q2n:STRATA_PF_FUSED_Q2N=1 fp16:STRATA_PREFILL_MMQ=0; do
  name=${arm%%:*}; envkv=${arm#*:}
  extra=(); [ -n "$envkv" ] && extra=(--env "$envkv")
  docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py run \
      /ab/cfg/ab-q2_0-static.json --exe "$E" --out "/ab/results/$RUN/kl/$name" --ids "$IDS" --fresh "$N" --xargs "--prompt-cache 0" "${extra[@]}" \
      > "$OUT/kl-$name.console.log" 2>&1
  echo "kl $name rc=$?"; docker rm strata-lab-ab-q2n > /dev/null 2>&1
  tail -1 "$OUT/kl-$name.console.log"
done
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/fp16" "/ab/results/$RUN/kl/base" "/ab/results/$RUN/kl/q2n" > "$OUT/kl-vs-fp16.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
docker run --name strata-lab-ab-q2n "${common[@]}" strata-lab-base /opt/strata/terminal/python/bin/python3 -I /ab/tools/kl_prompt.py compare \
    "/ab/results/$RUN/kl/base" "/ab/results/$RUN/kl/q2n" > "$OUT/kl-vs-mmq.txt" 2>&1; docker rm strata-lab-ab-q2n > /dev/null 2>&1
grep -h 'vs ' "$OUT/kl-vs-fp16.txt" "$OUT/kl-vs-mmq.txt"
nvidia-smi > "$OUT/nvidia-smi-after.txt"
echo "== $(date +%T) $RUN done"
