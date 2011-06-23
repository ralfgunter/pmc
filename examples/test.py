import pypmc

print("criar objeto")
obj = pypmc.PyPMC()

print("ExecConfig")
obj.n_threads = 4096
obj.n_iterations = 4
obj.rand_seed = 32898232

print("n_photons")
obj.n_photons = 2**22

print("detectors")
obj.detectors = [[(75, 35, 10), 1]]

print("tissues")
obj.tissues = [(1.0 / 1.1, 0.01, 0.05, 1.0)]

print("grid_dimensions")
obj.grid_dimensions = ((250, 10.0), (70, 10.0), (60, 10.0))

print("time_params")
obj.time_params = (0, 5e-9, 5e-9)

print("load_medium")
obj.load_medium("MouseLimb.bin", 250, 70, 60)

print("fluence_box")
obj.fluence_box = ((0, 249), (0, 69), (0, 59))

print("src_dir")
obj.src_dir = (0.0, 0.0, 1.0)

print("src_pos")
obj.src_pos = (12.5, 3.5, 1.0)

print("push_parameters")
obj.push_parameters()

print("run_simulation")
obj.run_simulation()

print("pull_results")
obj.pull_results()

print("write_to_disk")
obj.write_to_disk("out")
