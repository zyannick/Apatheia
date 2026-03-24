#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(apatheia, m) {
    m.doc() = "Apatheia: GPU-accelerated differentiable rigid body physics engine";
}
