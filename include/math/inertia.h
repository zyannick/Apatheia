#pragma once

#include <cuda_runtime.h>

#include "mat3x3.h"
#include "quaternion.h"
#include "vec3.h"

namespace apatheia {

template<typename T>
struct Inertia {
    T mass;
    Mat3x3<T> Ibody;
    Mat3x3<T> IbodyInv;

    __host__ __device__ Inertia()
        : mass(T(1)), Ibody(Mat3x3<T>::identity()), IbodyInv(Mat3x3<T>::identity()) {}

    __host__ __device__ Inertia(T m, const Mat3x3<T>& I)
        : mass(m), Ibody(I), IbodyInv(I.inverse()) {}

    __host__ __device__ static Inertia fromPrincipalMoments(T m, T Ixx, T Iyy, T Izz) {
        return Inertia(m, Mat3x3<T>::diag(Ixx, Iyy, Izz));
    }

    __host__ __device__ static Inertia sphere(T m, T r) {
        T I = (T(2) / T(5)) * m * r * r;
        return fromPrincipalMoments(m, I, I, I);
    }

    __host__ __device__ static Inertia box(T m, T hx, T hy, T hz) {
        T Ixx = (T(1) / T(12)) * m * (T(4) * hy * hy + T(4) * hz * hz);
        T Iyy = (T(1) / T(12)) * m * (T(4) * hx * hx + T(4) * hz * hz);
        T Izz = (T(1) / T(12)) * m * (T(4) * hx * hx + T(4) * hy * hy);
        return fromPrincipalMoments(m, Ixx, Iyy, Izz);
    }

    __host__ __device__ static Inertia cylinder(T m, T r, T h) {
        T Ixx = (T(1) / T(12)) * m * (T(3) * r * r + T(4) * h * h);
        T Izz = T(0.5) * m * r * r;
        return fromPrincipalMoments(m, Ixx, Ixx, Izz);
    }

    __host__ __device__ Mat3x3<T> worldInertia(const Quaternion<T>& q) const {
        Mat3x3<T> R = q.toRotationMatrix();
        return R * Ibody * R.transposed();
    }

    __host__ __device__ Mat3x3<T> worldInertiaInv(const Quaternion<T>& q) const {
        Mat3x3<T> R = q.toRotationMatrix();
        return R * IbodyInv * R.transposed();
    }

    __host__ __device__ T effectiveMass(const Vec3<T>& r, const Vec3<T>& n,
                                        const Mat3x3<T>& Iinv) const {
        Vec3<T> rxn = r.cross(n);
        return T(1) / mass + dot(rxn, Iinv * rxn);
    }
};

using Inertiaf = Inertia<float>;

}  // namespace apatheia
