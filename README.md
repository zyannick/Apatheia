# APATHEIA

> A GPU-accelerated, differentiable rigid body physics engine — built from scratch in pure C++/CUDA.

The name comes from the Stoic concept of *apatheia*: freedom from disturbance. A fitting goal for a physics engine where numerical stability is a first-class design constraint.

---

## What it is

Apatheia is a rigid body simulator designed for high-throughput reinforcement learning and robotics research. It runs entirely on the GPU, exposes full gradients through the simulation step, and is built to benchmark against [Isaac Lab](https://developer.nvidia.com/isaac/lab) and [Brax](https://github.com/google/brax).

**Key properties:**
- Zero Python in the hot path — pure C++/CUDA
- Differentiable end-to-end via adjoint method
- Numerically stable by design (semi-implicit integration, quaternion guard rails)
- Race-free parallel contact resolution via mass splitting

---

## Architecture

```
apatheia/
├── include/
│   ├── math/
│   ├── core/
│   └── gpu/
├── src/
│   ├── core/
│   └── gpu/
├── tests/
└── benchmarks/
```

### Layers

| Layer | Components |
|---|---|
| Public API | `World`, `Body`, `step()`, `grad_step()` |
| Core engine | Simulation loop · Constraint solver · Grad engine |
| GPU kernels | Body · Collision BVH · Contact · Grad |
| Math primitives | Vec3/Mat3x3 · Quaternion · Inertia tensor · Dual numbers |

---

## Technical design

### Integration
Semi-implicit Euler with the velocity-then-position update order. Quaternion dynamics use the exponential map to stay on SO(3) without drift. Normalization is enforced every step.

### Constraint solver
XPBD with parallel island decomposition. Contacts are resolved using mass splitting (Tonge et al., 2012) to eliminate race conditions across concurrent constraint updates — no atomic locks in the hot path.

### Collision detection
- **Broad phase**: Linear BVH (LBVH) built on the GPU each frame using Morton codes
- **Narrow phase**: GJK for convex–convex pairs

### Differentiability
`grad_step()` records a forward tape during integration, then runs the adjoint method backward through the full step. Gradients flow through forces, constraint impulses, and contact resolution. Dual-number forward-mode AD in the math primitives enables gradient checking during development.

---

## Status

- [x] Math primitives (Vec3, Quaternion, Inertia, DualNumber)
- [ ] Single-body CPU prototype
- [ ] CUDA body integration kernel
- [ ] LBVH collision pipeline
- [ ] Parallel constraint solver
- [ ] Adjoint grad engine
- [ ] Benchmarks vs Isaac Gym / Brax

---

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**Requirements:** CUDA 12+, C++17, CMake 3.24+

---

## References

- Baraff (1997) — *An Introduction to Physically Based Modeling: Rigid Body Simulation*
- Tonge et al. (2012) — *Mass splitting for jitter-free parallel rigid body simulation*
- de Avila Belbute-Peres et al. (NeurIPS 2018) — *End-to-End Differentiable Physics for Learning and Control*
- Hu et al. (ICLR 2020) — *DiffTaichi: Differentiable Programming for Physical Simulation*
- Makoviychuk et al. (NeurIPS 2021) — *Isaac Gym: High Performance GPU-Based Physics Simulation*

---

## License

MIT