import pypmc

max_time = 1e-10
step     = 0.1e-10
num_steps = int(max_time / step)

obj = pypmc.PyPMC()

obj.n_threads = 4096 
obj.n_iterations = 2
obj.rand_seed = 32898232

obj.n_photons = 2**21

obj.detectors = [(73, 33, 8, 1)]

obj.tissues = [(1.1, 0.01, 0.05, 1.0)]

obj.grid_dimensions = ((250, 0.1), (70, 0.1), (60, 0.1))

obj.time_params = (0, max_time, step)

obj.load_medium("MouseLimb.bin", 250, 70, 60)

obj.fluence_box = ((0, 249), (0, 69), (0, 59))

obj.src_dir = (0.0, 0.0, 1.0)

obj.src_pos = (12.5, 3.5, 1.0)

obj.run_simulation()

obj.pull_results()

print("Fluence results:")
for i in range(num_steps):
    print("t" + repr(i) + " = " + repr(obj.fluence[73][33][12][i]))
