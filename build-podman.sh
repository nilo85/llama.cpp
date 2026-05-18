#!/bin/sh

#sudo podman build --format docker -t llama.cpp:full-intel --build-arg ONEAPI_VERSION=2026.0.0-devel-ubuntu24.04 --build-arg GGML_SYCL_F16=ON --build-arg GGML_SYCL_DEVICE_ARCH=bmg  --target full -f .devops/intel.Dockerfile .
# sudo podman build --format docker -t llama.cpp:full-intel --build-arg GGML_SYCL_F16=ON --build-arg GGML_SYCL_DEVICE_ARCH=bmg  --target full -f .devops/intel.Dockerfile .


sudo podman build --format docker -t llama.cpp:full-intel-extras \
  --build-arg GGML_SYCL_F16=ON \
  --build-arg GGML_SYCL_DEVICE_ARCH=bmg \
  --target full -f .devops/intel.Dockerfile .

#  --build-arg EXTRA_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-L/opt/intel/oneapi/compiler/latest/linux/bin-llvm" \
  #  --build-arg ONEAPI_VERSION=2026.0.0-devel-ubuntu24.04 \
