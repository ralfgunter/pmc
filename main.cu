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
* Example:                                                                      *
*         tMCimg input.inp                                                      *
*                                                                               *
* Please find more details in README and doc/HELP                               *
********************************************************************************/

#include "main.h"

int main(int argc, char *argv[])
{
    if (argc != 4) {
        printf( "usage: tMCimg input_file.inp threads_per_block n_threads\n" );
        exit(1);
    }

    ExecConfig conf;
    Simulation sim;
    GPUMemory gmem;

    // Parse .inp file into the simulation structure.
    read_input(&conf, &sim, argv[1]);

    // TODO: put these somewhere else, and explain why the numbers are such
    int n = atoi(argv[2]);
    int n_threads = atoi(argv[3]);
    conf.n_threads_per_block = n;
    conf.n_threads = n_threads;
    conf.n_blocks = conf.n_threads / conf.n_threads_per_block;
    printf("#blocks = %d\nthreads per block = %d\n", conf.n_blocks, conf.n_threads_per_block);
    printf("#photons = %d\n#threads = %d\n", sim.n_photons, conf.n_threads);
    if(conf.rand_seed > 0)
        srand(conf.rand_seed);
    else
        srand(time(NULL));
    correct_source(&sim);

    // Allocate and initialize memory to be used by the GPU.
    printf("Ready to copy a bunch of data!\n");
    init_mem(conf, &sim, &gmem);

    // Run simulations on the GPU.
    printf("Ready to do some neat stuff!\n");
    simulate(conf, sim, gmem);

    // Retrieve results to host
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
