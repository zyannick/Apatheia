#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace apatheia {

template<typename T>
struct Vec3 {
    T x, y, z;

    __host__ __device__ Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    __host__ __device__ explicit Vec3(T s) : x(s), y(s), z(s) {}
    __host__ __device__ Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ static Vec3 zero() { return Vec3(T(0), T(0), T(0)); }
    __host__ __device__ static Vec3 one() { return Vec3(T(1), T(1), T(1)); }
    __host__ __device__ static Vec3 unitX() { return Vec3(T(1), T(0), T(0)); }
    __host__ __device__ static Vec3 unitY() { return Vec3(T(0), T(1), T(0)); }
    __host__ __device__ static Vec3 unitZ() { return Vec3(T(0), T(0), T(1)); }

    __host__ __device__ T& operator[](int i) { return (&x)[i]; }
    __host__ __device__ T operator[](int i) const { return (&x)[i]; }

    __host__ __device__ Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    __host__ __device__ Vec3 operator*(T s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(T s) const {
        T r = T(1) / s;
        return {x * r, y * r, z * r};
    }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    __host__ __device__ Vec3& operator-=(const Vec3& v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    __host__ __device__ Vec3& operator*=(T s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
    __host__ __device__ Vec3& operator/=(T s) {
        T r = T(1) / s;
        x *= r;
        y *= r;
        z *= r;
        return *this;
    }

    __host__ __device__ T dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }

    __host__ __device__ Vec3 cross(const Vec3& v) const {
        return {y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
    }

    __host__ __device__ T normSq() const { return x * x + y * y + z * z; }

    __host__ __device__ T norm() const {
        using std::sqrt;
        return sqrt(normSq());
    }

    __host__ __device__ Vec3 normalized() const {
        T n = norm();
        return (n > T(1e-10)) ? (*this / n) : Vec3::zero();
    }

    __host__ __device__ Vec3 componentWiseMul(const Vec3& v) const {
        return {x * v.x, y * v.y, z * v.z};
    }
    __host__ __device__ Vec3 componentWiseDiv(const Vec3& v) const {
        return {x / v.x, y / v.y, z / v.z};
    }

    __host__ __device__ T maxComponent() const {
        T m = (x > y) ? x : y;
        return (m > z) ? m : z;
    }
    __host__ __device__ T minComponent() const {
        T m = (x < y) ? x : y;
        return (m < z) ? m : z;
    }
};

template<typename T>
__host__ __device__ inline Vec3<T> operator*(T s, const Vec3<T>& v) {
    return v * s;
}

template<typename T>
__host__ __device__ inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
    return a.dot(b);
}
template<typename T>
__host__ __device__ inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b) {
    return a.cross(b);
}
template<typename T>
__host__ __device__ inline T norm(const Vec3<T>& v) {
    return v.norm();
}
template<typename T>
__host__ __device__ inline Vec3<T> normalize(const Vec3<T>& v) {
    return v.normalized();
}

using Vec3f = Vec3<float>;

}  // namespace apatheia
