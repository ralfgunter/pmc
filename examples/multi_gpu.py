import pypmc
import numpy

max_time = 5e-10
step     = 1e-10
num_steps = int(max_time / step)

# GPU number 1
gpu1 = pypmc.PyPMC()
gpu1.n_threads = 16384
gpu1.n_iterations = 1
gpu1.rand_seed = 32898232
gpu1.n_photons = 2**24
gpu1.detectors = [(74, 34, 9, 1)]
gpu1.tissues = [(1.1, 0.01, 0.05, 1.0)]
gpu1.grid_dimensions = ((250, 0.1), (70, 0.1), (60, 0.1))
gpu1.time_params = (0, max_time, step)
gpu1.load_medium("MouseLimb.bin", 250, 70, 60)
gpu1.fluence_box = ((0, 249), (0, 69), (0, 59))
gpu1.src_dir = (0.0, 0.0, 1.0)
gpu1.src_pos = (12.5, 3.5, 1.0)

# GPU number 2
gpu2 = pypmc.PyPMC()
gpu2.n_threads = 16384 
gpu2.n_iterations = 1
gpu2.rand_seed = 32898232
gpu2.n_photons = 2**24
gpu2.detectors = [(74, 34, 9, 1)]
gpu2.tissues = [(1.3, 0.01, 0.05, 1.0)]
gpu2.grid_dimensions = ((250, 0.1), (70, 0.1), (60, 0.1))
gpu2.time_params = (0, max_time, step)
gpu2.load_medium("MouseLimb.bin", 250, 70, 60)
gpu2.fluence_box = ((0, 249), (0, 69), (0, 59))
gpu2.src_dir = (0.0, 0.0, 1.0)
gpu2.src_pos = (12.5, 3.5, 1.0)

# First run
print("\nRun 1")
gpu1.run_simulation(0)

# Second run
print("\nRun 2")
gpu2.run_simulation(1)

# Pulling results
print("Pull 1")
gpu1.pull_results()
print("Pull 2")
gpu2.pull_results()

# Compare fluence from each run
print("\nFluence results: ")
for i in range(num_steps):
    print("gpu1: " + repr(gpu1.fluence[74][34][13][i]))
    print("gpu2: " + repr(gpu2.fluence[74][34][13][i]))
