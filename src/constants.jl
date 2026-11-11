const N_THREADS = 256
const pi_f32 = convert(Float32, pi)
const z_0 = ComplexF32(1.0+0.0im)
const z_90cw = ComplexF32(0.0-1.0im)
const z_90ccw = ComplexF32(0.0+1.0im)

# The legacy module globals `cdev` / `gdev` were removed: `gdev` baked a
# single target device in at precompile time. Use `select_device(backend)`,
# `cpu_device()`, or `to_device_of(ref, x)` (device follows the data).