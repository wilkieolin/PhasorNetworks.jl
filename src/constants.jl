const N_THREADS = 256
pi_f32 = convert(Float32, pi)

# Define devices
cdev = cpu_device()
if CUDA.functional()
    gdev = gpu_device()
end