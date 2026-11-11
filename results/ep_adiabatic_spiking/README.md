# `grid_readout_pilot.csv`

Pilot for the spike-timing readout axis: `ε` (extended down to 3e-4) × `ω_p` ×
`n_cycles` × `readout_δ`, at 784→256→64, hard projection, `K_mode=:zero`.

2570 of 2688 rows — the `ω_p=0.005, n_cycles=8` corner is missing because the
run was stopped to fix a scheduling bug (`Threads.@threads` static-schedules
contiguous chunks, and the deliberate cheapest-first job sort therefore handed
one thread the entire expensive tail; now `:dynamic`).

Findings live in `../ep_readout_floor/FINDINGS.md` — this file is the raw
pilot the first two results there are computed from.

Rows predate the `jitter`, `kmode` and `project` columns, so `report()` fills
them from defaults (`0.0`, `zero`, `hard`), which is what they in fact were.
