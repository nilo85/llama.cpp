#!/bin/sh

sudo podman build --format docker -t llama.cpp:full-intel --build-arg ONEAPI_VERSION=2026.0.0-devel-ubuntu24.04 --build-arg GGML_SYCL_F16=ON --build-arg GGML_SYCL_DEVICE_ARCH=bmg  --target full -f .devops/intel.Dockerfile .
