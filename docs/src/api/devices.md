# Device Cards API

A device card describes one hardware platform's phase unit (an oscillator
with its emitter) in physical units, with every value tagged by where it
comes from: measured, simulated, vendor, projected, fitted, our data,
derived, assumed, not yet checked (TK) or unknown. These are the tags of the
ARC H2 card template.

Cards are data only. Turning a card into simulation parameters, and running
networks on it, is a later layer of the device support; `docs/devices.md` sets
out the shared-ω ideal, what real devices do, and the conventions those
layers use.

A card is a TOML file with `schema = "phasor-device-card/1"` and sections
`identity`, `unit`, `spread`, `noise`, `orbit`, `emission`, `coupling`,
`trim`, `clock`, `reader` and `cost`. Optional `[operating_points.<name>]`
tables override fields, for example for a device that is a driven resonator
below threshold and a limit-cycle oscillator above it:

```toml
schema = "phasor-device-card/1"

[identity]
name  = { value = "VO2 relaxation oscillator", type = "assumed" }
class = { value = "R", type = "assumed" }

[unit]
f0_Hz = { value = 3.0e6, type = "meas", cond = "300 K", src = "doi:..." }
Q     = { type = "unknown" }

[orbit]
G_table = { file = "vo2_G.csv", type = "sim", src = "..." }

[operating_points.above_threshold]
identity = { class = { value = "L", type = "assumed" } }
```

```@docs
DEVICE_CARD_SCHEMA
DEVICE_CARD_FIELDS
DeviceCard
CardValue
CardIssue
load_device_card
parse_device_card
validate_device_card
card_value
card_class
```

## Device specs

[`device_spec`](@ref) turns a card into a [`DeviceSpec`](@ref): the same
device in dimensionless simulation units (`docs/devices.md` C1). Times are in
the device's own period `1/f₀`, rates are multiplied by it, and frequencies
are relative to `ω₀`, so one spec runs at any `spk_args.t_period`. Every
sub-spec defaults to the shared-ω ideal, so a spec only describes how the
device departs from it, and `is_ideal(IdealDevice())` holds.

```julia
card = load_device_card("cards/vo2.toml")
spec = device_spec(card, SpikingArgs(); op = :above_threshold)
spec.unknown      # fields the card marks unknown, replaced by the ideal
spec.provenance   # each card field used, with its value type
spec.notes        # table fits and their residuals, unit conversions

# Or by hand, e.g. for a sweep:
spec = DeviceSpec(spread = SpreadSpec(sigma_omega = 0.02),
                  trim = TrimSpec(freq = :quantised, freq_bits = 4))
```

`set = Dict("section.field" => value)` overrides card values before
conversion, in the card's units, which is how a sweep varies a value the
card does not know. Orbit tables become a [`FourierFn`](@ref) by default
(smooth, differentiable, GPU-friendly). Their fit residual goes into
`spec.notes`. Pass `tables = :table` for exact [`TableFn`](@ref) interpolation
instead.

```@docs
DeviceSpec
IdealDevice
device_spec
is_ideal
UnitSpec
SpreadSpec
OrbitSpec
NoiseSpec
EmitSpec
SynapseSpec
WireSpec
TrimSpec
ClockSpec
ReaderSpec
CostSpec
```

### Orbit functions

```@docs
PeriodicFn
FourierFn
TableFn
nharmonics
fourier_fit
speed_distortion
kick_distortion
mean_inverse_speed
```

### Non-isochrony

```@docs
AmpFn
amp_beta
beta_from_nu
beta_from_e4
beta_from_iso
```

### Phase noise

```@docs
diffusion_from_linewidth
diffusion_from_phase_noise
diffusion_from_jitter
```
