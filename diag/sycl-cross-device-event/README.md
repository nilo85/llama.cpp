# sycl cross-device event probe

Decisive Phase 5 test (us_sycl_starve.md F34): can a SYCL queue depend on a
foreign device's event? Result on 2x Arc Pro B70, oneAPI 2026.1.1:
- both default queues share ONE context: YES
- same-device depends_on: OK
- cross-device depends_on: ABORT (SIGABRT in level_zero cmdlist_hw.inl)

So the cross-device event_synchronize in the split loop cannot be made device-side on this driver. Version-dependent: re-run on a newer toolchain.

Build + run (JIT, oneAPI 2026.1.1, both GPUs exposed):
```
source /opt/intel/oneapi/setvars.sh
icpx -fsycl -O2 sycl_ctx_test.cpp -o sycl_ctx_test
ZES_ENABLE_SYSMAN=1 ./sycl_ctx_test   # needs renderD129 + renderD130
```
