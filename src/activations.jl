# Layer activations operating in the complex domain.
#
# These functions are intended to be passed as the `activation` argument to
# layers like `PhasorDense`, `PhasorConv`, `PhasorResonant`, `ResonantSTFT`,
# `PhasorFixed`, and the `SSM*Attention` family. They take a complex-valued
# input (typically the post-`W·x + bias` linear output of a layer) and return
# a complex-valued output suitable for the next stage.
#
# Pure domain conversions (`angle_to_complex`, `complex_to_angle`,
# `cmpx_to_realvec`, `realvec_to_cmpx`, spike helpers, etc.) live in
# `src/domains.jl`.

"""
    soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)

Soft angle-only activation: shrink the angle of each complex input toward 0
based on its magnitude, then re-emit on the unit circle.

For each `x`, computes a sigmoid blend `s ∈ [0, 1]` of `|x|` over the
transition window `[r_lo, r_hi]` and returns `exp(i · s · angle(x))` —
unit modulus, with phase scaled by `s`. Sub-threshold inputs (`|x| ≪ r_lo`)
get `s ≈ 0` and collapse to `1 + 0im` (silent reference); supra-threshold
inputs (`|x| ≫ r_hi`) preserve their phase on the unit circle.

# Arguments
- `x::AbstractArray{<:Complex}`: Array of complex numbers.
- `r_lo::Real = 0.1f0`: Lower threshold for magnitude scaling.
- `r_hi::Real = 0.2f0`: Upper threshold for magnitude scaling.

# Returns
- `AbstractArray{<:Complex}` on the unit circle: `exp(i · s · angle(x))`.

See also: [`soft_normalize_to_unit_circle`](@ref), which is the SLERP-style
twin (interpolates phase between `0` and `angle(z)` rather than scaling it).
"""
function soft_angle(x::AbstractArray{<:Complex}, r_lo::Real = 0.1f0, r_hi::Real = 0.2f0)
    s = similar(real.(x))

    ignore_derivatives() do
        r = abs.(x)
        m = (r .- r_lo) ./ (r_hi - r_lo)
        s .= sigmoid_fast(3.0f0 .* m .- (r_hi - r_lo))
    end

    θ = s .* angle.(x)
    return cos.(θ) .+ im .* sin.(θ)
end


"""
    normalize_to_unit_circle(z::AbstractArray{<:Complex}; ε = 1f-8, threshold = 1f-10)

Project complex numbers (approximately) onto the unit circle. Two modes,
selected by `ε`:

- `ε > 0` (default, **safe**): `y = z ./ sqrt(real(z)^2 + imag(z)^2 + ε)`.
  The denominator is bounded below by `√ε`, so both forward and backward
  are smooth everywhere — no custom `rrule` needed, no NaN from the
  `abs(z)` chain rule at exact `z = 0`. Output magnitude is `≈ 1` for
  `|z| ≫ √ε` and approaches `0` for `|z| ≪ √ε` (silent input ⇒ silent
  output).
- `ε = 0` (**hard**): `y = z ./ |z|` with a `1 + 0im` fallback for
  `|z| ≤ threshold`. Output magnitude is exactly 1 for in-band inputs.
  This branch dispatches into a separate kernel that carries a
  closed-form `rrule` returning zero cotangent for sub-threshold
  elements (so `abs(0)` chain rule cannot produce `0·NaN = NaN`).

# Arguments
- `z::AbstractArray{<:Complex}`: complex input array.
- `ε::Real = 1.0f-8`: safe-mode regularization. Set `ε = 0` to recover
  the hard `z/|z|` projection. The transition from `magnitude ≈ 1` to
  `magnitude ≈ |z|/√ε` happens around `|z| ≈ √ε ≈ 1e-4`. The Jacobian
  magnitude near `z = 0` scales as `1/√ε`.
- `threshold::Real = 1.0f-10`: only used when `ε = 0`; sub-threshold
  inputs map to `1 + 0im` and receive zero cotangent.

# Comparison

| `|z|`            | `ε = 1f-8` (default)            | `ε = 0` (hard)              |
|------------------|---------------------------------|-----------------------------|
| `≫ √ε` (typical) | `≈ z / |z|`, magnitude `≈ 1`    | `z / |z|`, magnitude 1      |
| `= √ε`           | `z / √(2ε)`, magnitude `1/√2`   | `z / |z|`, magnitude 1      |
| `= 0`            | `0 + 0im`                       | `1 + 0im` (fallback)        |

For a soft, magnitude-preserving variant that smoothly interpolates the
*phase* between sub-threshold and supra-threshold regions, see
[`soft_normalize_to_unit_circle`](@ref).
"""
function normalize_to_unit_circle(z::AbstractArray{<:Complex};
                                   ε::Real = 1.0f-8,
                                   threshold::Real = 1.0f-10)
    if Float32(ε) == 0.0f0
        return _normalize_to_unit_circle_hard(z; threshold = threshold)
    else
        return z ./ sqrt.(abs2.(real.(z)) .+ abs2.(imag.(z)) .+ Float32(ε))
    end
end

# Hard projection helper: z/|z| with a `1+0im` sub-threshold fallback.
# Carries a closed-form rrule (below) so the backward at z=0 is finite.
# Not exported — reach it via `normalize_to_unit_circle(z; ε = 0)`.
function _normalize_to_unit_circle_hard(z::AbstractArray{<:Complex};
                                         threshold::Real = 1.0f-10)
    th = Float32(threshold)
    r = abs.(z)
    safe_r = max.(r, th)
    unit_z = z ./ safe_r
    default_value = ComplexF32(1.0f0 + 0.0f0im)
    return ifelse.(r .> th, unit_z, default_value)
end

# Closed-form pullback for the hard branch. For y = z/r the real-pair
# Jacobian collapses, in ChainRules tangent convention `ȳ = ∂L/∂(re y) +
# i·∂L/∂(im y)`, to
#
#     dz = -i · z · imag(z · conj(ȳ)) / r³
#
# Returning a hard zero on sub-threshold elements stops the upstream
# `abs(z)` chain rule from contributing `(z̄ / |z|) · dr = NaN` at z = 0
# and poisoning the rest of the gradient via `0 · NaN = NaN` (IEEE 754).
function ChainRulesCore.rrule(::typeof(_normalize_to_unit_circle_hard),
                              z::AbstractArray{<:Complex};
                              threshold::Real = 1.0f-10)
    th = Float32(threshold)
    r = abs.(z)
    safe_r = max.(r, th)
    unit_z = z ./ safe_r
    default_value = ComplexF32(1.0f0 + 0.0f0im)
    y = ifelse.(r .> th, unit_z, default_value)

    function _normalize_to_unit_circle_hard_pullback(ȳ_)
        ȳ = unthunk(ȳ_)
        active = r .> th
        dz_active = (-1.0f0im) .* z .* imag.(z .* conj.(ȳ)) ./ (safe_r .^ 3)
        dz = ifelse.(active, dz_active, zero(eltype(z)))
        return (NoTangent(), dz)
    end
    return y, _normalize_to_unit_circle_hard_pullback
end

"""
    soft_normalize_to_unit_circle(z::AbstractArray{<:Complex}; r_lo::Real = 0.1f0, r_hi::Real = 0.6f0, steepness::Float32 = 10.0f0)

Soft normalization: smoothly map complex numbers onto the unit circle via phase interpolation
(SLERP), with sub-threshold values collapsed to `1+0im`.

# Arguments
- `z::AbstractArray{<:Complex}`: Array of complex numbers
- `r_lo::Real`: Lower magnitude threshold — below this, output is approximately `1+0im` (default: 0.1)
- `r_hi::Real`: Upper magnitude threshold — above this, output is approximately `z/|z|` (default: 0.6)

# Returns
- Complex array with magnitude exactly 1, with phase smoothly interpolated from 0 toward `angle(z)`

# Details
Uses spherical linear interpolation (SLERP) on the unit circle rather than linear complex mixing,
which avoids cancellation artifacts when `z` and the reference `1+0im` point in opposite directions.

A blend factor in [0, 1] is computed via sigmoid of the input magnitude, centered between `r_lo`
and `r_hi`. The output phase is then `blend × angle(z)`:
- For |z| ≪ r_lo: blend ≈ 0 → output ≈ `1+0im` (sub-threshold / silent)
- For |z| ≫ r_hi: blend ≈ 1 → output ≈ `z/|z|` (suprathreshold / active)
- For r_lo ≤ |z| ≤ r_hi: smooth phase rotation between the two extremes

The output magnitude is always exactly 1, making this suitable as a differentiable gate that
preserves phase information for active neurons while anchoring silent neurons at a fixed reference.

# Example
```julia
# Sub-threshold: output collapses to 1+0im regardless of phase
z_small = 0.05f0 * exp(1im * π/4)
z_norm = soft_normalize_to_unit_circle([z_small])  # ≈ 1+0im, |z_norm| = 1

# Suprathreshold: output preserves phase on unit circle
z_large = 5.0f0 * exp(1im * π/4)
z_norm = soft_normalize_to_unit_circle([z_large])  # ≈ exp(iπ/4), |z_norm| = 1
```
"""
function soft_normalize_to_unit_circle(z::AbstractArray{<:Complex}; r_lo::Real = 0.1f0, r_hi::Real = 0.6f0)
    r = abs.(z)
    midpoint = (r_lo + r_hi) / 2
    k = 6.0f0 / (r_hi - r_lo)
    # blend: 0 when r << r_lo (→ output 1+0im), 1 when r >> r_hi (→ output z/|z|)
    blend = sigmoid_fast.(k .* (r .- midpoint))

    # Unit vector in direction of z
    safe_r = max.(r, 1.0f-10)
    unit_z = z ./ safe_r

    # Interpolate phase from 0 toward angle(z) — no complex addition, no cancellation
    θ = atan.(imag.(unit_z), real.(unit_z))
    return cos.(blend .* θ) .+ im .* sin.(blend .* θ)
end

"""
    soft_normalize_to_unit_circle(z, r_lo, r_hi)

Positional-argument form for use with **trainable thresholds**. `r_lo` and
`r_hi` may be scalars or arrays broadcasting with `z` (e.g. a length-`C`
vector reshaped to `(C, 1, 1)` to apply a per-channel gate over `(C, L, B)`
data). Semantics match the kwargs form; the difference is that the
companion `rrule` for this method also propagates gradients into `r_lo`
and `r_hi`, so they can sit in a Lux layer's `parameters` and be trained
end-to-end.

For the math and stability rationale see the kwargs version above.
"""
function soft_normalize_to_unit_circle(z::AbstractArray{<:Complex},
                                       r_lo::Union{Real, AbstractArray},
                                       r_hi::Union{Real, AbstractArray})
    rl = Float32.(r_lo)
    rh = Float32.(r_hi)
    midpoint = (rl .+ rh) ./ 2f0
    k = 6f0 ./ (rh .- rl)
    r = abs.(z)
    safe_r = max.(r, 1f-10)
    blend = sigmoid_fast.(k .* (r .- midpoint))
    unit_z = z ./ safe_r
    θ = atan.(imag.(unit_z), real.(unit_z))
    return cos.(blend .* θ) .+ im .* sin.(blend .* θ)
end

# Closed-form pullback for the positional (trainable-threshold) form.
# Forward: y = exp(i·φ),  φ = blend(r, r_lo, r_hi) · θ(z),
#   blend = σ(u),  u = k · (r − m),  k = 6/(r_hi − r_lo),  m = (r_lo + r_hi)/2
#   θ     = atan(imag(z/safe_r), real(z/safe_r))
#
# dz follows the kwargs-version derivation (k σ' is ∂blend/∂r).
# For the thresholds:
#   ∂k/∂r_lo =  k² / 6,   ∂k/∂r_hi = −k² / 6
#   ∂m/∂r_lo =  ∂m/∂r_hi = 0.5
#   ∂u/∂r_lo =  (k²/6)·(r−m) − 0.5·k
#   ∂u/∂r_hi = −(k²/6)·(r−m) − 0.5·k
# and ∂φ/∂r_• = θ · σ'(u) · ∂u/∂r_•.
# Per-element contributions are reduced (summed) to match the shapes of
# `r_lo` and `r_hi` — handles both scalar thresholds (sum over all elements)
# and broadcasted per-channel vectors (sum over the broadcast-1 dims).
function ChainRulesCore.rrule(::typeof(soft_normalize_to_unit_circle),
                              z::AbstractArray{<:Complex},
                              r_lo::Union{Real, AbstractArray},
                              r_hi::Union{Real, AbstractArray})
    rl = Float32.(r_lo)
    rh = Float32.(r_hi)
    midpoint = (rl .+ rh) ./ 2f0
    k = 6f0 ./ (rh .- rl)
    th = 1f-10

    r = abs.(z)
    safe_r = max.(r, th)
    u = k .* (r .- midpoint)
    blend = sigmoid_fast.(u)
    unit_z = z ./ safe_r
    θ = atan.(imag.(unit_z), real.(unit_z))
    y = cos.(blend .* θ) .+ im .* sin.(blend .* θ)

    function soft_normalize_to_unit_circle_pos_pullback(ȳ_)
        ȳ = unthunk(ȳ_)
        active = r .> th
        dLdφ = imag.(ȳ .* conj.(y))                          # real
        σ_deriv = blend .* (1f0 .- blend)                    # σ'(u)

        # ---- dz (matches the kwargs-version derivation) ----
        blend_deriv_r = k .* σ_deriv                         # ∂blend/∂r
        coeff = (θ .* blend_deriv_r .* safe_r .+ 1f0im .* blend) ./ (safe_r .^ 2)
        dz_active = dLdφ .* z .* coeff
        dz = ifelse.(active, dz_active, zero(eltype(z)))

        # ---- dr_lo, dr_hi ----
        kk_over_6 = (k .* k) ./ 6f0
        gap_term  = kk_over_6 .* (r .- midpoint)             # (k²/6)·(r−m)
        du_drlo   =   gap_term .- 0.5f0 .* k
        du_drhi   =  -gap_term .- 0.5f0 .* k
        common    = ifelse.(active, dLdφ .* σ_deriv .* θ, zero(eltype(blend)))
        per_drlo  = common .* du_drlo
        per_drhi  = common .* du_drhi

        dr_lo = _reduce_to_shape(per_drlo, r_lo)
        dr_hi = _reduce_to_shape(per_drhi, r_hi)
        return (NoTangent(), dz, dr_lo, dr_hi)
    end
    return y, soft_normalize_to_unit_circle_pos_pullback
end

# Reduce a per-element gradient array to the shape of `target`, by summing
# over dimensions on which `target` is a singleton (or trailing). Handles
# both scalar (Real) and array targets — scalars get the full sum.
_reduce_to_shape(grad::AbstractArray, ::Real) = sum(grad)
function _reduce_to_shape(grad::AbstractArray, target::AbstractArray)
    dims_to_sum = Tuple(i for i in 1:ndims(grad)
                         if i > ndims(target) || size(target, i) == 1)
    result = isempty(dims_to_sum) ? grad : sum(grad; dims=dims_to_sum)
    if ndims(target) < ndims(result)
        result = dropdims(result; dims=Tuple(ndims(target)+1:ndims(result)))
    end
    return result
end

# Closed-form pullback for soft_normalize_to_unit_circle.
#
# Forward: y = exp(i · φ),  φ = blend(r) · θ(z),
#   blend = σ(k · (r − m)),  k = 6/(r_hi − r_lo),  m = (r_lo + r_hi)/2
#   θ     = atan(imag(z), real(z))
#
# Real-cost cotangent in ChainRules convention (`ȳ = ∂L/∂(re y) + i·∂L/∂(im y)`):
#   ∂L/∂φ = imag(ȳ · conj(y))
#   ∂φ/∂a + i·∂φ/∂b = z · (θ · blend' · r + i · blend) / r²,  blend' = k · blend · (1 − blend)
# Hence:
#   dz = imag(ȳ · conj(y)) · z · (θ · blend' · r + i · blend) / r²    for r > th
#   dz = 0                                                           for r ≤ th
#
# The sub-threshold zero short-circuits the upstream `abs(z)` and `atan(b,a)`
# chain rules, both of which produce NaN at z = 0 and would otherwise poison
# the rest of the gradient via `0 · NaN = NaN`.
function ChainRulesCore.rrule(::typeof(soft_normalize_to_unit_circle),
                              z::AbstractArray{<:Complex};
                              r_lo::Real = 0.1f0, r_hi::Real = 0.6f0)
    rl = Float32(r_lo); rh = Float32(r_hi)
    midpoint = (rl + rh) / 2f0
    k = 6f0 / (rh - rl)
    th = 1f-10                        # safe-divisor floor (matches forward)

    r = abs.(z)
    safe_r = max.(r, th)
    blend = sigmoid_fast.(k .* (r .- midpoint))
    unit_z = z ./ safe_r
    θ = atan.(imag.(unit_z), real.(unit_z))
    y = cos.(blend .* θ) .+ im .* sin.(blend .* θ)

    function soft_normalize_to_unit_circle_pullback(ȳ_)
        ȳ = unthunk(ȳ_)
        active = r .> th
        dLdφ = imag.(ȳ .* conj.(y))                          # real
        blend_deriv = k .* blend .* (1f0 .- blend)           # real
        # coeff bounded everywhere because we use safe_r (≥ th)
        coeff = (θ .* blend_deriv .* safe_r .+ 1f0im .* blend) ./ (safe_r .^ 2)
        dz_active = dLdφ .* z .* coeff
        dz = ifelse.(active, dz_active, zero(eltype(z)))
        return (NoTangent(), dz)
    end
    return y, soft_normalize_to_unit_circle_pullback
end
