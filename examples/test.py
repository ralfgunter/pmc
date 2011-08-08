import pypmc
import numpy

max_time = 5e-10
step     = 1e-10
num_steps = int(max_time / step)

obj = pypmc.PyPMC()

obj.n_threads = 4096 
obj.n_iterations = 1
obj.rand_seed = 32898232

obj.n_photons = 2**20
obj.detectors = [(74, 34, 9, 1)]
obj.tissues = [(1.1, 0.01, 0.05, 1.0)]
obj.grid_dimensions = ((250, 0.1), (70, 0.1), (60, 0.1))
obj.time_params = (0, max_time, step)
obj.load_medium("MouseLimb.bin", 250, 70, 60)
obj.fluence_box = ((0, 249), (0, 69), (0, 59))
obj.src_dir = (0.0, 0.0, 1.0)
obj.src_pos = (12.5, 3.5, 1.0)

# First run
print("Running the first simulation")
obj.run_simulation(0)
obj.sync()

# Copy first run's fluence to another array.
f = numpy.asfortranarray(obj.fluence).copy()

# Second run
print("Running the second simulation")
obj.n_photons = 2**21
obj.run_simulation(0)
obj.sync()

# Compare fluence from each run
for i in range(num_steps):
    print(f[74][34][13][i])
    print(obj.fluence[74][34][13][i])
