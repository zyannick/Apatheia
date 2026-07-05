// Week 2 exit criteria (portion 1/2): Vec3, Mat3x3, and Inertia closed forms.
// Quaternion-specific tests (orthogonality under repeated composition,
// rotate()/toRotationMatrix() consistency, exp/log round-trip) live in
// test_quaternion.cpp.

#include <gtest/gtest.h>

#include "math/math.h"

using namespace apatheia;

namespace {
constexpr float kEps = 1e-5f;

void expectMatNear(const Mat3x3f& A, const Mat3x3f& B, float eps = kEps) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(A(i, j), B(i, j), eps) << "at (" << i << "," << j << ")";
}
}  // namespace

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

TEST(Vec3Test, DotAndCrossBasis) {
    Vec3f x = Vec3f::unitX();
    Vec3f y = Vec3f::unitY();
    Vec3f z = Vec3f::unitZ();

    EXPECT_NEAR(dot(x, y), 0.0f, kEps);
    EXPECT_NEAR(dot(x, x), 1.0f, kEps);

    Vec3f xy = cross(x, y);
    EXPECT_NEAR(xy.x, z.x, kEps);
    EXPECT_NEAR(xy.y, z.y, kEps);
    EXPECT_NEAR(xy.z, z.z, kEps);
}

TEST(Vec3Test, NormAndNormalize) {
    Vec3f v(3.0f, 4.0f, 0.0f);
    EXPECT_NEAR(v.norm(), 5.0f, kEps);

    Vec3f n = v.normalized();
    EXPECT_NEAR(n.norm(), 1.0f, kEps);
    EXPECT_NEAR(n.x, 0.6f, kEps);
    EXPECT_NEAR(n.y, 0.8f, kEps);
}

TEST(Vec3Test, ArithmeticOperators) {
    Vec3f a(1, 2, 3), b(4, 5, 6);

    Vec3f sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 5);
    EXPECT_FLOAT_EQ(sum.y, 7);
    EXPECT_FLOAT_EQ(sum.z, 9);

    Vec3f scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x, 2);
    EXPECT_FLOAT_EQ(scaled.y, 4);
    EXPECT_FLOAT_EQ(scaled.z, 6);

    Vec3f diff = b - a;
    EXPECT_FLOAT_EQ(diff.x, 3);
    EXPECT_FLOAT_EQ(diff.y, 3);
    EXPECT_FLOAT_EQ(diff.z, 3);
}

// ---------------------------------------------------------------------------
// Mat3x3
// ---------------------------------------------------------------------------

TEST(Mat3x3Test, IdentityIsMultiplicativeIdentity) {
    Mat3x3f M = Mat3x3f::fromRows(Vec3f(2, 1, 0), Vec3f(0, 3, 1), Vec3f(1, 0, 4));
    Mat3x3f I = Mat3x3f::identity();
    expectMatNear(M * I, M);
    expectMatNear(I * M, M);
}

TEST(Mat3x3Test, InverseTimesSelfIsIdentity) {
    Mat3x3f M = Mat3x3f::fromRows(Vec3f(2, 1, 0), Vec3f(0, 3, 1), Vec3f(1, 0, 4));
    Mat3x3f inv = M.inverse();
    expectMatNear(M * inv, Mat3x3f::identity(), 1e-4f);
    expectMatNear(inv * M, Mat3x3f::identity(), 1e-4f);
}

TEST(Mat3x3Test, TransposeReversesIndices) {
    Mat3x3f M = Mat3x3f::fromRows(Vec3f(1, 2, 3), Vec3f(4, 5, 6), Vec3f(7, 8, 9));
    Mat3x3f Mt = M.transposed();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(Mt(i, j), M(j, i));
}

TEST(Mat3x3Test, SkewMatchesCrossProduct) {
    Vec3f v(1, 2, 3);
    Vec3f w(4, -1, 2);
    Mat3x3f Vx = Mat3x3f::skew(v);

    Vec3f viaSkew = Vx * w;
    Vec3f viaCross = cross(v, w);

    EXPECT_NEAR(viaSkew.x, viaCross.x, kEps);
    EXPECT_NEAR(viaSkew.y, viaCross.y, kEps);
    EXPECT_NEAR(viaSkew.z, viaCross.z, kEps);
}

TEST(Mat3x3Test, OuterProductStructure) {
    Vec3f a(1, 2, 3), b(4, 5, 6);
    Mat3x3f M = Mat3x3f::outer(a, b);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) EXPECT_FLOAT_EQ(M(i, j), a[i] * b[j]);
}

TEST(Mat3x3Test, DeterminantAndTraceKnownValues) {
    Mat3x3f D = Mat3x3f::diag(2, 3, 4);
    EXPECT_FLOAT_EQ(D.det(), 24.0f);
    EXPECT_FLOAT_EQ(D.trace(), 9.0f);
}

// ---------------------------------------------------------------------------
// Inertia — closed forms + parallel-axis theorem
// ---------------------------------------------------------------------------

TEST(InertiaTest, SphereClosedForm) {
    float m = 2.0f, r = 3.0f;
    Inertia<float> sph = Inertia<float>::sphere(m, r);
    float expected = (2.0f / 5.0f) * m * r * r;

    EXPECT_NEAR(sph.Ibody(0, 0), expected, kEps);
    EXPECT_NEAR(sph.Ibody(1, 1), expected, kEps);
    EXPECT_NEAR(sph.Ibody(2, 2), expected, kEps);

    // A sphere's inertia tensor is isotropic: off-diagonal terms vanish.
    EXPECT_NEAR(sph.Ibody(0, 1), 0.0f, kEps);
    EXPECT_NEAR(sph.Ibody(0, 2), 0.0f, kEps);
    EXPECT_NEAR(sph.Ibody(1, 2), 0.0f, kEps);
}

TEST(InertiaTest, BoxClosedForm) {
    float m = 6.0f, hx = 1.0f, hy = 2.0f, hz = 3.0f;
    Inertia<float> boxI = Inertia<float>::box(m, hx, hy, hz);

    float Ixx = (1.0f / 12.0f) * m * (4 * hy * hy + 4 * hz * hz);
    float Iyy = (1.0f / 12.0f) * m * (4 * hx * hx + 4 * hz * hz);
    float Izz = (1.0f / 12.0f) * m * (4 * hx * hx + 4 * hy * hy);

    EXPECT_NEAR(boxI.Ibody(0, 0), Ixx, kEps);
    EXPECT_NEAR(boxI.Ibody(1, 1), Iyy, kEps);
    EXPECT_NEAR(boxI.Ibody(2, 2), Izz, kEps);
}

// Parallel-axis theorem: I_about_axis = I_cm + m * (|d|^2 * Identity - d ⊗ d).
// For an offset purely along x by distance d, this adds m*d^2 to Iyy and Izz
// only, leaving Ixx untouched -- a simple, hand-verifiable case.
TEST(InertiaTest, ParallelAxisTheoremOffsetAlongX) {
    float m = 2.0f, r = 1.5f;
    Inertia<float> sph = Inertia<float>::sphere(m, r);

    float d = 4.0f;
    Vec3f offset(d, 0.0f, 0.0f);

    Mat3x3f shift = Mat3x3f::identity() * offset.normSq() - Mat3x3f::outer(offset, offset);
    Mat3x3f shifted = sph.Ibody + shift * m;

    float I0 = (2.0f / 5.0f) * m * r * r;
    EXPECT_NEAR(shifted(0, 0), I0, kEps);              // Ixx unchanged
    EXPECT_NEAR(shifted(1, 1), I0 + m * d * d, kEps);  // Iyy += m d^2
    EXPECT_NEAR(shifted(2, 2), I0 + m * d * d, kEps);  // Izz += m d^2
    EXPECT_NEAR(shifted(0, 1), 0.0f, kEps);
    EXPECT_NEAR(shifted(0, 2), 0.0f, kEps);
    EXPECT_NEAR(shifted(1, 2), 0.0f, kEps);
}

TEST(InertiaTest, IbodyInvIsPrecomputedInverse) {
    Inertia<float> boxI = Inertia<float>::box(3.0f, 1.0f, 2.0f, 0.5f);
    Mat3x3f prod = boxI.Ibody * boxI.IbodyInv;
    expectMatNear(prod, Mat3x3f::identity(), 1e-4f);
}
