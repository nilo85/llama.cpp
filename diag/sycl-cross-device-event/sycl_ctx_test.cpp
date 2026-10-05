// Verify SYCL event-dependency capabilities on the dual-B70 L0 driver.
//  - same-device depends_on (device N event -> device N queue): expected OK
//  - cross-device depends_on (device 0 event -> device 1 queue): expected ABORT
// This decides whether the cross-device event_synchronize can be made async.
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
using namespace sycl;

int main() {
    try {
        auto plat = platform();
        std::vector<device> all = plat.get_devices();
        std::vector<device> gpus;
        for (auto & d : all) if (d.is_gpu()) gpus.push_back(d);
        std::cout << "num devices total: " << all.size() << "  gpus: " << gpus.size() << std::endl;
        for (auto & d : gpus) std::cout << "  gpu: " << d.get_info<info::device::name>() << std::endl;
        if (gpus.size() < 2) { std::cout << "NEED 2 GPUS" << std::endl; return 2; }

        queue q0(gpus[0]);
        queue q1(gpus[1]);
        std::cout << "queue(gpu) same context: " << (q0.get_context() == q1.get_context() ? "YES" : "NO") << std::endl;

        int * b0 = malloc_device<int>(1, q0);
        int * b1 = malloc_device<int>(1, q1);

        // 1) same-device depends_on (device 0 event -> device 0 queue)
        event e0s = q0.parallel_for(range(1), [=](id<1> i) { b0[i[0]] = 1; });
        try {
            q0.submit([&](handler & h) { h.depends_on(e0s); h.parallel_for(range(1), [=](id<1> i) { b0[i[0]] = 2; }); });
            q0.wait_and_throw();
            std::cout << "SAME-DEVICE depends_on: OK" << std::endl;
        } catch (exception & e) {
            std::cout << "SAME-DEVICE depends_on FAILED: " << e.what() << std::endl;
        }

        // 2) cross-device depends_on (device 0 event -> device 1 queue): expect abort
        std::cout << "now testing CROSS-DEVICE depends_on (device0 event -> device1 queue)..." << std::endl;
        std::cout.flush();
        event e0 = q0.parallel_for(range(1), [=](id<1> i) { b0[i[0]] = 42; });
        q1.submit([&](handler & h) { h.depends_on(e0); h.parallel_for(range(1), [=](id<1> i) { b1[i[0]] = 7; }); });
        q1.wait_and_throw();
        std::cout << "CROSS-DEVICE depends_on: OK (unexpected!)" << std::endl;

        free(b0, q0);
        free(b1, q1);
        return 0;
    } catch (exception & e) {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
