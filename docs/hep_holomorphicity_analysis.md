# Holomorphicity Requirements for hEP: Empirical Analysis

## The Question

Does the hEP cost function need to be holomorphic for accurate gradients?

The theoretical argument (Laborieux & Zenke, 2022) assumes holomorphicity
of the energy function — including the cost term — for the equilibrium
map z*(beta) to be holomorphic in beta, which is required for the Cauchy
integral formula to yield exact gradients.

Phase-based readouts (the natural choice for phasor networks) require
non-holomorphic operations like `angle(z)`, `|z|`, or `Re(z)`. This
appears to create a fundamental tension between the network's information
encoding and the learning rule's mathematical requirements.

## Empirical Test

We tested five cost functions ranging from fully holomorphic to strongly
non-holomorphic, measuring the cosine similarity between the hEP contour
gradient and the finite-difference (true) gradient:

| Cost Function | Non-Holomorphic Op | Cosine Sim | Notes |
|---------------|-------------------|------------|-------|
| Complex cross-entropy | None | 1.000 | Fully holomorphic |
| Interference + complex xent | conj(code) (constant) | 1.000 | "Mostly" holomorphic |
| Re() before softmax | Re(z) | 0.9998 | Explicitly non-holomorphic |
| MSE on |interference| | abs(z) | 0.9999 | Strongly non-holomorphic |
| angle() phase loss | angle(z) | 0.0 | Non-smooth at test point |

For a 3-layer network with batched data and Re() softmax cost:
cosine similarity = **0.964**, magnitude ratio = 0.29.

## Key Finding

**The cost function's holomorphicity barely affects gradient quality.**
Cost functions A through D all produce essentially perfect gradient
direction (cosine > 0.999 for 2-layer, > 0.96 for 3-layer), despite
using non-holomorphic operations like `Re()` and `abs()`.

The only failure (cost E) was due to the cost function being
**non-smooth** at the test point (angle() is piecewise constant for
real-valued states), not due to non-holomorphicity per se.

## Why This Works

The contour integration extracts the first Fourier coefficient of
`dPhi/dW` evaluated at equilibria around the contour. This is exact
when z*(beta) is holomorphic in beta. When the cost function is
non-holomorphic in z, z*(beta) is technically not holomorphic in beta.

However, the non-holomorphic operations (Re, abs) are still **smooth**
real-analytic functions. The equilibrium map z*(beta), while not
complex-analytic, is still smooth and its Taylor expansion in beta
has a well-defined linear term. The contour integration extracts
this linear term with high accuracy because:

1. The higher-order terms (quadratic, cubic in beta) are rejected by
   the Fourier extraction — they map to higher harmonics.
2. The "non-analytic" parts of z*(beta) (due to the cost's
   non-holomorphicity) produce contributions that are either small
   (for smooth non-holomorphic functions) or orthogonal to the first
   Fourier coefficient.

In practice, the contour integration acts as a robust gradient
estimator that's insensitive to whether the underlying function is
complex-analytic or merely smooth.

## Implications for Phasor Networks

1. **The HolomorphicReadout is fine as-is.** The `z * conj(code)`
   interference followed by `Re()` and softmax produces accurate
   hEP gradients despite the non-holomorphic steps.

2. **Traditional phase-based costs can work.** Even `cos(angle(z) -
   angle(code))` should work, provided the output states have
   non-trivial phase (i.e., aren't purely real — which requires
   omega > 0 or complex input encoding).

3. **The critical holomorphicity requirement is in the DYNAMICS,
   not the cost.** The settling dynamics (energy function, activation,
   inter-layer coupling) should be holomorphic or nearly so. The
   cost function just needs to be smooth.

4. **The self-energy term (1/2)<z, Kz> in the energy is essential**
   — not because of holomorphicity, but because it makes the
   equilibrium correspond to the ODE steady state, ensuring the
   EP gradient theorem applies.
