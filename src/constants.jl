const N_THREADS = 256
const pi_f32 = convert(Float32, pi)
const z_0 = ComplexF32(1.0+0.0im)
const z_90cw = ComplexF32(0.0-1.0im)
const z_90ccw = ComplexF32(0.0+1.0im)

# Define devices (backward compat — new code should use select_device)
cdev = cpu_device()
gdev = CUDA.functional() ? gpu_device() : cpu_device()