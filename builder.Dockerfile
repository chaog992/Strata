# strata-lab-ab-builder: compiles only the engine binary ("strata" CMake target) of any tag, CUDA 13.0, sm_120.
# Same toolchain as the repo's Dockerfile at v0.1.44.1 / v0.1.45 / v0.2.2 (nvidia/cuda:13.0.0-devel-ubuntu24.04, gcc 13)
# and the cmake/ninja pins of requirements(-build).txt at those tags (cmake 4.4.3, ninja 1.13.2).
#   docker build -t strata-lab-ab-builder -f builder.Dockerfile .
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl unzip python3 python3-venv time \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir cmake==4.4.3 ninja==1.13.2
ENV PATH=/venv/bin:$PATH
