#!/bin/sh
# Build the engine binary of each named source tree with ONE set of CMake options (the release line of
# packaging/build_cuda_linux.sh at v0.2.2, narrowed to sm_120), so the arms differ only in their source.
#
#   build_engines.sh v0.1.44.1 v0.1.45 v0.2.2      (inside strata-lab-ab-builder, /ab mounted)
#   TESTS=ON TARGETS="strata prefill_fused_iq_test" build_engines.sh pf-q2n     (a tree with a GPU test as well)
#
# Expects: /ab/src/src-<TAG>/ (git archive of the tag), /ab/third_party/llama.cpp (llama.cpp at 3cf03257, the pin of all
# three tags: setup.py LLAMA_CPP_COMMIT and third_party/ggml/VERSION.txt).  Writes /ab/build/<TAG>/ and /ab/engines/<TAG>/.
# PORTABLE=OFF builds ggml-cpu for this CPU (GGML_NATIVE) instead of the AVX2 baseline the release packages use.
set -eu
J=${MAX_JOBS:-8}
PORTABLE=${PORTABLE:-ON}
TESTS=${TESTS:-OFF}                     # ON: the test targets can be built too (STRATA_BUILD_TESTS)
TARGETS=${TARGETS:-strata}              # the CMake targets to build, copied to /ab/engines/<TAG>/
LLAMA=${LLAMA:-/ab/third_party/llama.cpp}
SUFFIX=${SUFFIX:-}                      # e.g. SUFFIX=-native with PORTABLE=OFF, so both variants can coexist
[ -f "$LLAMA/ggml/CMakeLists.txt" ] || { echo "no llama.cpp at $LLAMA" >&2; exit 2; }
mkdir -p /ab/cudalibs
# one copy of the CUDA 13.0 runtime libraries for every arm (the engine links cudart, cublas(+Lt) and the driver's libcuda)
for l in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do
  cp -aL "/usr/local/cuda/lib64/$l" /ab/cudalibs/ 2>/dev/null || cp -aL /usr/local/cuda/targets/x86_64-linux/lib/$l /ab/cudalibs/
done
for tag in "$@"; do
  src=/ab/src/src-$tag; bld=/ab/build/$tag$SUFFIX; out=/ab/engines/$tag$SUFFIX
  [ -f "$src/CMakeLists.txt" ] || { echo "no source tree $src" >&2; exit 2; }
  mkdir -p "$bld" "$out"
  t0=$(date +%s)
  cmake -G Ninja -S "$src" -B "$bld" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS="$TESTS" \
      -DSTRATA_EXPERIMENTAL_SM60=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE="$PORTABLE" \
      -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
      "-DCMAKE_CUDA_ARCHITECTURES=120-real;120-virtual" -DSTRATA_GGML_DIR="$LLAMA" > "$out/configure.log" 2>&1
  for target in $TARGETS; do
    # setup.py's rule (#45): CUDA 13.0's ptxas now and then fails on a PTX file it just wrote; a second run gets past it
    if ! cmake --build "$bld" --target "$target" -j "$J" > "$out/build-$target.log" 2>&1; then
      echo "  ($tag: the build of $target stopped - trying it once more)"
      cmake --build "$bld" --target "$target" -j "$J" >> "$out/build-$target.log" 2>&1
    fi
    cp "$bld/$target" "$out/$target"
  done
  t1=$(date +%s)
  if [ -f "$bld/libstrata_kq.so" ]; then cp "$bld/libstrata_kq.so" "$out/"; fi
  (cd "$out" && sha256sum $TARGETS > SHA256SUMS)
  grep -E '^(CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_COMPILER_VERSION|STRATA_[A-Z0-9_]+|GGML_NATIVE|GGML_AVX[A-Z0-9_]*|GGML_FMA|GGML_F16C|GGML_BMI2)[:=]' \
      "$bld/CMakeCache.txt" > "$out/cmake-cache-flags.txt" || true
  nvcc --version | tail -2 > "$out/nvcc.txt"
  printf '{"tag": "%s", "portable": "%s", "jobs": %s, "seconds": %s, "targets": "%s"}\n' "$tag" "$PORTABLE" "$J" "$((t1 - t0))" "$TARGETS" > "$out/build-time.json"
  echo "built $tag$SUFFIX ($TARGETS) in $((t1 - t0)) s -> $out/"
done
