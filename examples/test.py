import pypmc

obj = pypmc.PyPMC()

obj.n_threads = 4096
obj.n_iterations = 8
obj.rand_seed = 32898232

obj.n_photons = 2**23

obj.detectors = [[(73, 33, 8), 1]]

obj.tissues = [(1.0 / 1.1, 0.01, 0.05, 1.0)]

obj.grid_dimensions = ((250, 10.0), (70, 10.0), (60, 10.0))

obj.time_params = (0, 5e-09, 5e-09)

obj.load_medium("MouseLimb.bin", 250, 70, 60)

obj.fluence_box = ((0, 249), (0, 69), (0, 59))

obj.src_dir = (0.0, 0.0, 1.0)

obj.src_pos = (12.5, 3.5, 1.0)

obj.run_simulation()

obj.pull_results()

obj.write_to_disk("out")
