import pypmc
import numpy

max_time = 5e-10
step     = 1e-10
num_steps = int(max_time / step)

obj = pypmc.PyPMC()

obj.n_threads = 16384
obj.n_iterations = 1
obj.rand_seed = 32898232

obj.n_photons = 2**20
obj.detectors = [(74, 34, 9, 1)]
obj.tissues = [(1.5, 0.018, 0.05, 1.0)]
obj.grid_dimensions = (0.1, 0.1, 0.1)
obj.time_params = (0, max_time, step)
obj.load_medium("MouseLimb.bin", 250, 70, 60)
obj.fluence_box = ((0, 249), (0, 69), (0, 59))
obj.srcs = [[(12.5, 3.5, 1.0), (0.0, 0.0, 1.0), 690], [(10.5, 3.5, 1.0), (0.0, 0.0, 1.0), 830]]

obj.run_simulation()

print obj.path_length
