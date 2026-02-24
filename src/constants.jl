const N_THREADS = 256
pi_f32 = convert(Float32, pi)
z_0 = ComplexF32(1.0+0.0im)
z_90cw = ComplexF32(0.0-1.0im)
z_90ccw = ComplexF32(0.0+1.0im)

# Define devices
cdev = cpu_device()
if CUDA.functional()
    gdev = gpu_device()
end