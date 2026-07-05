#pragma once

#include <cuda_runtime.h>

#include <cmath>

#include "mat3x3.h"
#include "vec3.h"

namespace apatheia {

template<typename T>
struct Quaternion {
    T w, x, y, z;

    __host__ __device__ Quaternion() : w(T(1)), x(T(0)), y(T(0)), z(T(0)) {}
    __host__ __device__ Quaternion(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}

    __host__ __device__ static Quaternion identity() { return Quaternion(T(1), T(0), T(0), T(0)); }

    __host__ __device__ static Quaternion fromAxisAngle(const Vec3<T>& axis, T angle) {
        using std::cos;
        using std::sin;
        T halfAngle = angle * T(0.5);
        T s         = sin(halfAngle);
        Vec3<T> n   = axis.normalized();
        return Quaternion(cos(halfAngle), n.x * s, n.y * s, n.z * s);
    }

    __host__ __device__ static Quaternion exp(const Vec3<T>& v) {
        using std::cos;
        using std::sin;
        T theta = v.norm();
        if (theta < T(1e-8)) {
            return Quaternion(T(1), T(0.5) * v.x, T(0.5) * v.y, T(0.5) * v.z).normalized();
        }
        T half = T(0.5) * theta;
        T s    = sin(half) / theta;
        return Quaternion(cos(half), v.x * s, v.y * s, v.z * s);
    }

    __host__ __device__ Vec3<T> log() const {
        using std::atan2;
        using std::sqrt;
        T n = sqrt(x * x + y * y + z * z);
        if (n < T(1e-8)) return Vec3<T>::zero();
        T theta = T(2) * atan2(n, w);
        T s     = theta / n;
        return Vec3<T>(x * s, y * s, z * s);
    }

    __host__ __device__ T normSq() const { return w * w + x * x + y * y + z * z; }
    __host__ __device__ T norm() const {
        using std::sqrt;
        return sqrt(normSq());
    }

    __host__ __device__ Quaternion normalized() const {
        T n = norm();
        return (n > T(1e-10)) ? Quaternion(w / n, x / n, y / n, z / n) : identity();
    }

    __host__ __device__ Quaternion canonical() const {
        return (w >= T(0)) ? *this : Quaternion(-w, -x, -y, -z);
    }

    __host__ __device__ Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }
    __host__ __device__ Quaternion inverse() const { return conjugate() * (T(1) / normSq()); }

    __host__ __device__ Quaternion operator*(const Quaternion& q) const {
        return Quaternion(
            w * q.w - x * q.x - y * q.y - z * q.z, w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x, w * q.z + x * q.y - y * q.x + z * q.w);
    }

    __host__ __device__ Quaternion operator*(T s) const {
        return Quaternion(w * s, x * s, y * s, z * s);
    }

    __host__ __device__ Quaternion operator+(const Quaternion& q) const {
        return Quaternion(w + q.w, x + q.x, y + q.y, z + q.z);
    }

    __host__ __device__ Vec3<T> rotate(const Vec3<T>& v) const {
        Vec3<T> qv(x, y, z);
        Vec3<T> t = T(2) * cross(qv, v);
        return v + w * t + cross(qv, t);
    }

    __host__ __device__ Mat3x3<T> toRotationMatrix() const {
        T xx = x * x, yy = y * y, zz = z * z;
        T xy = x * y, xz = x * z, yz = y * z;
        T wx = w * x, wy = w * y, wz = w * z;
        return Mat3x3<T>::fromRows(
            Vec3<T>(T(1) - T(2) * (yy + zz), T(2) * (xy - wz), T(2) * (xz + wy)),
            Vec3<T>(T(2) * (xy + wz), T(1) - T(2) * (xx + zz), T(2) * (yz - wx)),
            Vec3<T>(T(2) * (xz - wy), T(2) * (yz + wx), T(1) - T(2) * (xx + yy)));
    }

    /// Integrate body-frame angular velocity omega over dt using the
    /// exponential map (stays on SO(3) exactly, then re-normalized to
    /// absorb floating-point drift).
    __host__ __device__ Quaternion integrateAngularVelocity(const Vec3<T>& omega, T dt) const {
        // Quaternion::exp(v) already maps the *full* rotation vector
        // (axis * angle) to a quaternion, halving the angle internally
        // (exp(axis*angle) == fromAxisAngle(axis, angle)). Passing
        // 0.5*dt*omega here would halve the angle a second time,
        // rotating at half the correct angular rate.
        Quaternion dq = Quaternion::exp(dt * omega);
        return (*this * dq).normalized();
    }

    __host__ __device__ static Quaternion slerp(const Quaternion& a, const Quaternion& b, T t) {
        using std::abs;
        using std::acos;
        using std::cos;
        using std::sin;
        T d           = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
        Quaternion b2 = (d < T(0)) ? Quaternion(-b.w, -b.x, -b.y, -b.z) : b;
        d             = abs(d);
        if (d > T(0.9995)) {
            return Quaternion(a.w + t * (b2.w - a.w), a.x + t * (b2.x - a.x),
                              a.y + t * (b2.y - a.y), a.z + t * (b2.z - a.z))
                .normalized();
        }
        T theta_0 = acos(d);
        T theta   = theta_0 * t;
        T sin_t   = sin(theta);
        T sin_0   = sin(theta_0);
        T s0      = cos(theta) - d * sin_t / sin_0;
        T s1      = sin_t / sin_0;
        return Quaternion(s0 * a.w + s1 * b2.w, s0 * a.x + s1 * b2.x, s0 * a.y + s1 * b2.y,
                          s0 * a.z + s1 * b2.z);
    }
};

template<typename T>
__host__ __device__ inline Quaternion<T> operator*(T s, const Quaternion<T>& q) {
    return q * s;
}

using Quaternionf = Quaternion<float>;

}  // namespace apatheia
