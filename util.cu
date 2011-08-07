#include "main.h"

void linearize_3d(uint8_t ***t, uint8_t *l, int dim_x, int dim_y, int dim_z)
{
    for (int x = 0; x < dim_x; x++)
        for (int y = 0; y < dim_y; y++)
            for (int z = 0; z < dim_z; z++)
                l[LIN3D(x,y,z,dim_x,dim_y)] = t[x][y][z];
}


void parse_conf(ExecConfig *conf, int n_threads, int n_iterations)
{
    // Block dimension is fixed due to high register usage by the kernel.
    conf->n_threads = n_threads;
    conf->n_blocks = conf->n_threads / 128;
    conf->n_iterations = n_iterations;

    // For somewhat repeatable simulations, we allow the user to furnish a
    // positive seed that will eventually be used by the GPU RNG.
    // Negative seeds are ignored, and the system time is used instead.
    if(conf->rand_seed > 0)
        srand(conf->rand_seed);
    else
        srand(time(NULL));
}
