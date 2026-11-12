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
