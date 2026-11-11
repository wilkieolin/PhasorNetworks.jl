# Wave Sheets

Toroidal phasor sheets that carry phase waves between neighbouring cells, and
the analysis and hardware-modelling helpers built on them.

- [`PhasorWaveSheet`](@ref) is the coupled-oscillator surface layer. Its
  `transmit` mode selects what crosses a link: `:potential` (the internal
  potential), `:spike` (a soft unit phasor), or `:strict` (a unit phasor when
  `|z|` exceeds the threshold, otherwise nothing).
- [`ExcitableWaveSheet`](@ref) and [`soliton_simulate`](@ref) model excitable
  and soliton-like media.
- [`PhasorVelocityBank`](@ref) is a bank of sheets tuned to different
  conduction speeds, with hardware non-idealities in `velocity_bank_hw.jl`.
- Wave experts (`moe_gate`, `route_stats`, `update_moe_bias`) route patches
  to experts on the sheet.

See `docs/wave_dispersion_derivation.md` and `docs/wavesheet_experts_design.md`
for background.

## Wave sheet

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/wave.jl"]
```

## Excitable media

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/excitable.jl"]
```

## Velocity bank

```@autodocs
Modules = [PhasorNetworks]
Pages = ["src/velocity_bank.jl", "src/velocity_bank_hw.jl"]
```
