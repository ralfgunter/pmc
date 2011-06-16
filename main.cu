/********************************************************************************
*          Monte-Carlo Simulation for Light Transport in 3D Volumes             *
*********************************************************************************
*                                                                               *
* Copyright (C) 2002-2008,  David Boas    (dboas <at> nmr.mgh.harvard.edu)      *
*               2008        Jay Dubb      (jdubb <at> nmr.mgh.harvard.edu)      *
*               2008        Qianqian Fang (fangq <at> nmr.mgh.harvard.edu)      *
*               2011        Ralf Gunter   (ralfgunter <at> gmail.com)           *
*                                                                               *
* License:  4-clause BSD License, see LICENSE for details                       *
*                                                                               *
********************************************************************************/

#include "main.h"

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf( "usage: %s input_file.inp n_threads n_iterations\n", argv[0]);
        exit(1);
    }

    ExecConfig conf;
    Simulation sim;
    GPUMemory gmem;

    // Parse .inp file into the simulation structure.
    read_input(&conf, &sim, argv[1]);

    int n_threads = atoi(argv[2]);
    int n_iterations = atoi(argv[3]);
    parse_conf(&conf, n_threads, n_iterations);

    printf("blocks = %d\nthreads = %d\n", conf.n_blocks, conf.n_threads);
    printf("photons = %d\n", sim.n_photons);

    // Make sure the source is at an interface.
    correct_source(&sim);

    // Allocate and initialize memory to be used by the GPU.
    printf("Ready to copy a bunch of data!\n");
    init_mem(conf, &sim, &gmem);

    // Run simulations on the GPU.
    printf("Ready to do some neat stuff!\n");
    simulate(conf, sim, gmem);

    // Retrieve results to host.
    printf("Now we send this stuff back to the host!\n");
    retrieve(&sim, &gmem);

    // Write results to disk.
    printf("We'd better save those bits to disk before we're gone!\n");
    write_results(sim, argv[1]);

    // Clean up used memory.
    printf("Done! Now we clean up after ourselves.\n");
    free_mem(sim, gmem);

    return 0;
}
