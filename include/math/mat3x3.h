#pragma once

#include <cuda_runtime.h>

#include "vec3.h"

namespace apatheia {

template<typename T>
struct Mat3x3 {
    T m[3][3];

    __host__ __device__ Mat3x3() {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) m[i][j] = T(0);
    }

    __host__ __device__ static Mat3x3 identity() {
        Mat3x3 M;
        M.m[0][0] = M.m[1][1] = M.m[2][2] = T(1);
        return M;
    }

    __host__ __device__ static Mat3x3 zero() { return Mat3x3{}; }

    __host__ __device__ static Mat3x3 fromCols(const Vec3<T>& c0, const Vec3<T>& c1,
                                               const Vec3<T>& c2) {
        Mat3x3 M;
        M.m[0][0] = c0.x;
        M.m[1][0] = c0.y;
        M.m[2][0] = c0.z;
        M.m[0][1] = c1.x;
        M.m[1][1] = c1.y;
        M.m[2][1] = c1.z;
        M.m[0][2] = c2.x;
        M.m[1][2] = c2.y;
        M.m[2][2] = c2.z;
        return M;
    }

    __host__ __device__ static Mat3x3 fromRows(const Vec3<T>& r0, const Vec3<T>& r1,
                                               const Vec3<T>& r2) {
        Mat3x3 M;
        M.m[0][0] = r0.x;
        M.m[0][1] = r0.y;
        M.m[0][2] = r0.z;
        M.m[1][0] = r1.x;
        M.m[1][1] = r1.y;
        M.m[1][2] = r1.z;
        M.m[2][0] = r2.x;
        M.m[2][1] = r2.y;
        M.m[2][2] = r2.z;
        return M;
    }

    __host__ __device__ static Mat3x3 diag(T d0, T d1, T d2) {
        Mat3x3 M;
        M.m[0][0] = d0;
        M.m[1][1] = d1;
        M.m[2][2] = d2;
        return M;
    }

    __host__ __device__ T operator()(int r, int c) const { return m[r][c]; }
    __host__ __device__ T& operator()(int r, int c) { return m[r][c]; }

    __host__ __device__ Mat3x3 operator+(const Mat3x3& B) const {
        Mat3x3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) C.m[i][j] = m[i][j] + B.m[i][j];
        return C;
    }

    __host__ __device__ Mat3x3 operator-(const Mat3x3& B) const {
        Mat3x3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) C.m[i][j] = m[i][j] - B.m[i][j];
        return C;
    }

    __host__ __device__ Mat3x3 operator*(T s) const {
        Mat3x3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) C.m[i][j] = m[i][j] * s;
        return C;
    }

    __host__ __device__ Vec3<T> operator*(const Vec3<T>& v) const {
        return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
    }

    __host__ __device__ Mat3x3 operator*(const Mat3x3& B) const {
        Mat3x3 C;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                T s = T(0);
                for (int k = 0; k < 3; ++k) s += m[i][k] * B.m[k][j];
                C.m[i][j] = s;
            }
        return C;
    }

    __host__ __device__ Mat3x3 transposed() const {
        Mat3x3 result;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) result.m[i][j] = m[j][i];
        return result;
    }

    __host__ __device__ T trace() const { return m[0][0] + m[1][1] + m[2][2]; }

    __host__ __device__ T det() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
               m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
               m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }

    __host__ __device__ Mat3x3 inverse() const {
        T d = det();
        Mat3x3 inv;
        inv.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
        inv.m[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) / d;
        inv.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
        inv.m[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) / d;
        inv.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
        inv.m[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) / d;
        inv.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
        inv.m[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) / d;
        inv.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
        return inv;
    }

    __host__ __device__ static Mat3x3 outer(const Vec3<T>& a, const Vec3<T>& b) {
        Mat3x3 M;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M.m[i][j] = a[i] * b[j];
        return M;
    }

    __host__ __device__ static Mat3x3 skew(const Vec3<T>& v) {
        Mat3x3 M;
        M.m[0][1] = -v.z;
        M.m[0][2] = v.y;
        M.m[1][0] = v.z;
        M.m[1][2] = -v.x;
        M.m[2][0] = -v.y;
        M.m[2][1] = v.x;
        return M;
    }
};

template<typename T>
__host__ __device__ inline Mat3x3<T> operator*(T s, const Mat3x3<T>& M) {
    return M * s;
}

using Mat3x3f = Mat3x3<float>;

}  // namespace apatheia
