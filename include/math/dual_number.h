#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace apatheia {

template<typename T = float>
struct Dual {
    T real;
    T dual;

    __host__ __device__ Dual() : real(T(0)), dual(T(0)) {}
    __host__ __device__ explicit Dual(T r) : real(r), dual(T(0)) {}
    __host__ __device__ Dual(T r, T d) : real(r), dual(d) {}

    __host__ __device__ static Dual var(T a) { return Dual(a, T(1)); }
    __host__ __device__ static Dual constant(T a) { return Dual(a, T(0)); }

    __host__ __device__ Dual operator+(const Dual& b) const {
        return {real + b.real, dual + b.dual};
    }
    __host__ __device__ Dual operator-(const Dual& b) const {
        return {real - b.real, dual - b.dual};
    }
    __host__ __device__ Dual operator-() const { return {-real, -dual}; }

    __host__ __device__ Dual operator*(const Dual& b) const {
        return {real * b.real, real * b.dual + dual * b.real};
    }

    __host__ __device__ Dual operator/(const Dual& b) const {
        T inv = T(1) / b.real;
        return {real * inv, (dual * b.real - real * b.dual) * (inv * inv)};
    }

    __host__ __device__ Dual operator+(T s) const { return {real + s, dual}; }
    __host__ __device__ Dual operator-(T s) const { return {real - s, dual}; }
    __host__ __device__ Dual operator*(T s) const { return {real * s, dual * s}; }
    __host__ __device__ Dual operator/(T s) const {
        T inv = T(1) / s;
        return {real * inv, dual * inv};
    }

    __host__ __device__ Dual& operator+=(const Dual& b) {
        real += b.real;
        dual += b.dual;
        return *this;
    }
    __host__ __device__ Dual& operator-=(const Dual& b) {
        real -= b.real;
        dual -= b.dual;
        return *this;
    }
    __host__ __device__ Dual& operator*=(const Dual& b) {
        T newDual = real * b.dual + dual * b.real;
        real *= b.real;
        dual = newDual;
        return *this;
    }
    __host__ __device__ Dual& operator/=(const Dual& b) {
        T inv     = T(1) / b.real;
        T newDual = (dual * b.real - real * b.dual) * (inv * inv);
        real *= inv;
        dual = newDual;
        return *this;
    }

    __host__ __device__ Dual& operator+=(T s) {
        real += s;
        return *this;
    }
    __host__ __device__ Dual& operator-=(T s) {
        real -= s;
        return *this;
    }
    __host__ __device__ Dual& operator*=(T s) {
        real *= s;
        dual *= s;
        return *this;
    }
    __host__ __device__ Dual& operator/=(T s) {
        T inv = T(1) / s;
        real *= inv;
        dual *= inv;
        return *this;
    }

    __host__ __device__ bool operator<(const Dual& b) const { return real < b.real; }
    __host__ __device__ bool operator>(const Dual& b) const { return real > b.real; }
};

template<typename T>
__host__ __device__ inline Dual<T> operator+(T s, const Dual<T>& d) {
    return {s + d.real, d.dual};
}
template<typename T>
__host__ __device__ inline Dual<T> operator-(T s, const Dual<T>& d) {
    return {s - d.real, -d.dual};
}
template<typename T>
__host__ __device__ inline Dual<T> operator*(T s, const Dual<T>& d) {
    return {s * d.real, s * d.dual};
}
template<typename T>
__host__ __device__ inline Dual<T> operator/(T s, const Dual<T>& d) {
    T inv = T(1) / d.real;
    return {s * inv, -s * d.dual * (inv * inv)};
}

template<typename T>
__host__ __device__ inline Dual<T> sqrt(const Dual<T>& d) {
    using std::sqrt;
    T r = sqrt(d.real);
    return {r, d.dual * (T(0.5) / r)};
}

template<typename T>
__host__ __device__ inline Dual<T> sin(const Dual<T>& d) {
    using std::cos;
    using std::sin;
    return {sin(d.real), d.dual * cos(d.real)};
}

template<typename T>
__host__ __device__ inline Dual<T> cos(const Dual<T>& d) {
    using std::cos;
    using std::sin;
    return {cos(d.real), -d.dual * sin(d.real)};
}

template<typename T>
__host__ __device__ inline Dual<T> exp(const Dual<T>& d) {
    using std::exp;
    T e = exp(d.real);
    return {e, d.dual * e};
}

template<typename T>
__host__ __device__ inline Dual<T> log(const Dual<T>& d) {
    using std::log;
    return {log(d.real), d.dual / d.real};
}

template<typename T>
__host__ __device__ inline Dual<T> abs(const Dual<T>& d) {
    using std::abs;
    T sign = (d.real >= T(0)) ? T(1) : T(-1);
    return {abs(d.real), d.dual * sign};
}

template<typename T>
__host__ __device__ inline Dual<T> atan2(const Dual<T>& y, const Dual<T>& x) {
    using std::atan2;
    T denom = x.real * x.real + y.real * y.real;
    return {atan2(y.real, x.real), (x.real * y.dual - y.real * x.dual) / denom};
}

template<typename T>
__host__ __device__ inline Dual<T> acos(const Dual<T>& d) {
    using std::acos;
    using std::sqrt;
    T r = acos(d.real);
    return {r, -d.dual / sqrt(T(1) - d.real * d.real)};
}

template<typename T>
__host__ __device__ inline Dual<T> max(const Dual<T>& a, const Dual<T>& b) {
    return (a.real >= b.real) ? a : b;
}

template<typename T>
__host__ __device__ inline Dual<T> min(const Dual<T>& a, const Dual<T>& b) {
    return (a.real <= b.real) ? a : b;
}

using DualF = Dual<float>;

}  // namespace apatheia
