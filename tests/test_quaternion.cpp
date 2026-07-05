// Week 2 exit criteria (portion 2/2): quaternion orthogonality under
// repeated composition, rotate()/toRotationMatrix() round-trip
// consistency, and exp/log inverse behavior.

#include <gtest/gtest.h>

#include <cmath>

#include "math/math.h"

using namespace apatheia;

namespace {
constexpr float kEps = 1e-4f;
constexpr float kPi = 3.14159265358979323846f;

void expectMatNear(const Mat3x3f& A, const Mat3x3f& B, float eps = kEps) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(A(i, j), B(i, j), eps) << "at (" << i << "," << j << ")";
}

void expectQuatNear(const Quaternionf& a, const Quaternionf& b, float eps = kEps) {
    // q and -q represent the same rotation -- compare canonical forms
    // (w >= 0) so sign ambiguity doesn't cause spurious failures.
    Quaternionf ca = a.canonical();
    Quaternionf cb = b.canonical();
    EXPECT_NEAR(ca.w, cb.w, eps);
    EXPECT_NEAR(ca.x, cb.x, eps);
    EXPECT_NEAR(ca.y, cb.y, eps);
    EXPECT_NEAR(ca.z, cb.z, eps);
}
}  // namespace

TEST(QuaternionTest, IdentityRotationLeavesVectorUnchanged) {
    Quaternionf q = Quaternionf::identity();
    Vec3f v(1, 2, 3);
    Vec3f r = q.rotate(v);
    EXPECT_NEAR(r.x, v.x, kEps);
    EXPECT_NEAR(r.y, v.y, kEps);
    EXPECT_NEAR(r.z, v.z, kEps);
}

TEST(QuaternionTest, NinetyDegreesAboutZRotatesXToY) {
    Quaternionf q = Quaternionf::fromAxisAngle(Vec3f::unitZ(), kPi / 2.0f);
    Vec3f r = q.rotate(Vec3f::unitX());
    EXPECT_NEAR(r.x, 0.0f, kEps);
    EXPECT_NEAR(r.y, 1.0f, kEps);
    EXPECT_NEAR(r.z, 0.0f, kEps);
}

// rotate() and toRotationMatrix() must agree for arbitrary quaternions --
// the practical form of "round-trips to/from rotation matrix": R(q)*v
// must equal q.rotate(v) exactly.
TEST(QuaternionTest, RotateMatchesToRotationMatrix) {
    Quaternionf q = Quaternionf::fromAxisAngle(Vec3f(1, 1, 1).normalized(), 0.73f);
    Mat3x3f R = q.toRotationMatrix();

    Vec3f v(0.3f, -1.2f, 2.5f);
    Vec3f viaQuat = q.rotate(v);
    Vec3f viaMat = R * v;

    EXPECT_NEAR(viaQuat.x, viaMat.x, kEps);
    EXPECT_NEAR(viaQuat.y, viaMat.y, kEps);
    EXPECT_NEAR(viaQuat.z, viaMat.z, kEps);
}

TEST(QuaternionTest, RotationMatrixIsOrthogonalWithUnitDeterminant) {
    Quaternionf q = Quaternionf::fromAxisAngle(Vec3f(0.2f, -0.4f, 0.9f).normalized(), 1.1f);
    Mat3x3f R = q.toRotationMatrix();
    expectMatNear(R * R.transposed(), Mat3x3f::identity());
    EXPECT_NEAR(R.det(), 1.0f, 1e-3f);
}

// Repeated composition: chain 10,000 small rotations together (as a naive
// integrator would) and confirm the resulting rotation matrix is still
// orthogonal to good precision -- i.e. per-step normalization actually
// prevents drift over long compositions instead of merely looking fine
// for a handful of steps.
TEST(QuaternionTest, StaysOrthogonalAfterRepeatedComposition) {
    Quaternionf q = Quaternionf::identity();
    Quaternionf step = Quaternionf::fromAxisAngle(Vec3f(0.4f, 0.3f, 0.1f).normalized(), 0.001f);

    constexpr int kIterations = 10000;
    for (int i = 0; i < kIterations; ++i) {
        q = (q * step).normalized();
    }

    Mat3x3f R = q.toRotationMatrix();
    expectMatNear(R * R.transposed(), Mat3x3f::identity(), 1e-4f);
    EXPECT_NEAR(R.det(), 1.0f, 1e-3f);
}

// exp/log must be exact inverses of one another for rotation vectors well
// inside the (-pi, pi) range where the map is one-to-one.
TEST(QuaternionTest, ExpLogRoundTrip) {
    Vec3f v(0.3f, -0.6f, 0.2f);
    Quaternionf q = Quaternionf::exp(v);
    Vec3f back = q.log();
    EXPECT_NEAR(back.x, v.x, kEps);
    EXPECT_NEAR(back.y, v.y, kEps);
    EXPECT_NEAR(back.z, v.z, kEps);
}

TEST(QuaternionTest, ExpMatchesFromAxisAngle) {
    Vec3f axis = Vec3f(1, -2, 0.5f).normalized();
    float angle = 0.85f;
    Quaternionf viaExp = Quaternionf::exp(axis * angle);
    Quaternionf viaAxisAngle = Quaternionf::fromAxisAngle(axis, angle);
    expectQuatNear(viaExp, viaAxisAngle);
}

// Regression test for the exact factor-of-2 bug found in Week 2 review:
// integrating a constant angular velocity omega over N steps of dt must
// match a single closed-form rotation of angle N*dt*|omega| -- not half
// that angle.
TEST(QuaternionTest, IntegrateAngularVelocityMatchesClosedFormRotation) {
    Vec3f axis = Vec3f::unitZ();
    float omegaMag = 1.0f;
    float dt = 0.01f;
    constexpr int kSteps = 100;

    Quaternionf q = Quaternionf::identity();
    for (int i = 0; i < kSteps; ++i) {
        q = q.integrateAngularVelocity(axis * omegaMag, dt);
    }

    float totalAngle = kSteps * dt * omegaMag;  // == 1.0 rad
    Quaternionf expected = Quaternionf::fromAxisAngle(axis, totalAngle);
    expectQuatNear(q, expected, 1e-3f);
}

TEST(QuaternionTest, SlerpEndpointsMatchInputs) {
    Quaternionf a = Quaternionf::identity();
    Quaternionf b = Quaternionf::fromAxisAngle(Vec3f::unitY(), kPi / 2.0f);

    Quaternionf atStart = Quaternionf::slerp(a, b, 0.0f);
    Quaternionf atEnd = Quaternionf::slerp(a, b, 1.0f);

    expectQuatNear(atStart, a);
    expectQuatNear(atEnd, b);
}
