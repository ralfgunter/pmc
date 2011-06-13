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

template <class type>
void linearize_3d(type ***t, type *l, int dim_x, int dim_y, int dim_z)
{
    for (int x = 0; x < dim_x; x++)
        for (int y = 0; y < dim_y; y++)
            for (int z = 0; z < dim_z; z++)
                l[LIN3D(x,y,z,dim_x,dim_y)] = t[x][y][z];
}

void init_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem)
{
    float *d_II;
    //float *d_lenTiss, *d_momTiss;
    //uint *d_detHit_matrix;
    uint *h_seed, *d_seed;
    uchar *h_linear_tissueType, *d_tissueType;
    //int4 *d_detLoc;
    float4 *d_tissueProp;
    size_t num_tissueArrays, num_II, sizeof_seed;

    // Calculate the total number of voxel elements.
    int grid_dim = sim->grid.dim.x * sim->grid.dim.y * sim->grid.dim.z;

    // Linearize tissueType so that it can be converted into a cudaArray
    // and later bound to a 3d texture.
    h_linear_tissueType = (uchar *) malloc(grid_dim * sizeof(uchar));
    linearize_3d(sim->grid.tissueType, h_linear_tissueType,
                 sim->grid.dim.x, sim->grid.dim.y, sim->grid.dim.z);

/*
    cudaArray *d_volumeArray = 0;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    const cudaExtent volumeSize = make_cudaExtent(sim->grid.dim.x,
                                                  sim->grid.dim.y,
                                                  sim->grid.dim.z);
    cutilSafeCall( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void*)h_linear_tissueType, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cutilSafeCall( cudaMemcpy3D(&copyParams) );

    // set texture parameters
    tissueType.normalized = false;   // access with normalized texture coordinates
    tissueType.filterMode = cudaFilterModeLinear;      // linear interpolation
    tissueType.addressMode[0] = cudaAddressModeWrap;   // wrap tissueTypeture coordinates
    tissueType.addressMode[1] = cudaAddressModeWrap;
    tissueType.addressMode[2] = cudaAddressModeWrap;

    // bind array to 3D texture
    cutilSafeCall( cudaBindTextureToArray(tissueType, d_volumeArray, channelDesc) );
*/

    // Setup the path length and momentum transfer arrays.
    //num_tissueArrays = (sim->tiss.num + 1) * sim->n_photons;
    //sim->lenTiss = (float *) calloc(num_tissueArrays, sizeof(float));
    //sim->momTiss = (float *) calloc(num_tissueArrays, sizeof(float));

    // Photon fluence.
    num_II = sim->grid.nIxyz * sim->max_time;
    sim->II = (float *) calloc(num_II, sizeof(float));

    // Bitset indicating which detectors (if any) were hit by which photons.
    //sim->detHit = bitset_new(sim->n_photons, sim->det.num);

    // Seed used by the RNG.
    sizeof_seed = conf.n_threads * RAND_SEED_LEN * sizeof(uint);
    h_seed = (uint *) malloc(sizeof_seed);
    for(int i = 0; i < conf.n_threads * RAND_SEED_LEN; i++)
        h_seed[i] = rand();

    // Allocate memory on the GPU global memory.
    // TODO: use constant memory where appropriate 
    //cudaMalloc((void **) &d_detLoc, sim->det.num * sizeof(int4));
    cudaMalloc((void **) &d_tissueProp, (sim->tiss.num + 1) * sizeof(float4));
    cudaMalloc((void **) &d_tissueType, grid_dim * sizeof(uchar));
    //cudaMalloc((void **) &d_lenTiss, num_tissueArrays * sizeof(float));
    //cudaMalloc((void **) &d_momTiss, num_tissueArrays * sizeof(float));
    cudaMalloc((void **) &d_II,      num_II           * sizeof(float));
    //cudaMalloc((void **) &d_detHit_matrix, bitset_size(sim->detHit) * sizeof(uint));
    cudaMalloc((void **) &d_seed, sizeof_seed);

    // Copy simulation memory to the GPU.
    //cudaMemcpyToSymbol("detLoc", sim->det.info, sim->det.num * sizeof(int4));
    //cudaMemcpyToSymbol("tissueProp", sim->tiss.prop, (sim->tiss.num + 1) * sizeof(float4));
    cudaMemcpyToSymbol("s", sim, sizeof(Simulation));
    //TO_DEVICE(d_detLoc, sim->det.info, sim->det.num * sizeof(int4));
    TO_DEVICE(d_tissueProp, sim->tiss.prop, (sim->tiss.num + 1) * sizeof(float4));
    TO_DEVICE(d_tissueType, h_linear_tissueType, grid_dim * sizeof(uchar));
    //TO_DEVICE(d_lenTiss, sim->lenTiss, num_tissueArrays * sizeof(float));
    //TO_DEVICE(d_momTiss, sim->momTiss, num_tissueArrays * sizeof(float));
    TO_DEVICE(d_II,      sim->II,      num_II           * sizeof(float));
    //TO_DEVICE(d_detHit_matrix, sim->detHit.matrix, bitset_size(sim->detHit) * sizeof(uint));
    TO_DEVICE(d_seed, h_seed, sizeof_seed);

    // Update GPU memory structure (so that its pointers can be used elsewhere).
    //gmem->detLoc = d_detLoc;
    gmem->tissueProp = d_tissueProp;
    gmem->tissueType = d_tissueType;
    //gmem->lenTiss = d_lenTiss;
    //gmem->momTiss = d_momTiss;
    gmem->II = d_II;
    //gmem->detHit = sim->detHit;
    //gmem->detHit.matrix = d_detHit_matrix;
    gmem->seed = d_seed;
    cudaMemcpyToSymbol("g", gmem, sizeof(GPUMemory));

    // Free temporary memory used on the host.
    free(h_linear_tissueType);
    free(h_seed);
}

void free_mem(Simulation sim, GPUMemory gmem)
{
    // Tissue types.
    for(int i = 0; i < sim.grid.dim.x; i++) {
        for(int j = 0; j < sim.grid.dim.y; j++) {
            free(sim.grid.tissueType[i][j]);
        }
        free(sim.grid.tissueType[i]);
    }
    free(sim.grid.tissueType);
    cudaFree(gmem.tissueType);
    //cudaUnbindTexture(tissueType);

    // Detectors' locations and radii.
    //free(sim.det.info);
    //cudaFree(gmem.detLoc);

    // Optical properties of the different tissue types.
    free(sim.tiss.prop);
    cudaFree(gmem.tissueProp);

    // Path length and momentum transfer.
    //free(sim.lenTiss);
    //free(sim.momTiss);
    //cudaFree(gmem.lenTiss);
    //cudaFree(gmem.momTiss);

    // Photon fluence.
    free(sim.II);
    cudaFree(gmem.II);

    // Random number generation.
    cudaFree(gmem.seed);

    // Bitset of the detectors which were hit by a given photon.
    //bitset_free(sim.detHit);
    //cudaFree(gmem.detHit.matrix);  // TODO: properly handle this 
}

void retrieve(Simulation *sim, GPUMemory *gmem)
{
    size_t sizeof_tissueArrays = sim->n_photons * (sim->tiss.num + 1) * sizeof(float);
    size_t sizeof_II = sim->grid.nIxyz * sim->max_time * sizeof(float);
    size_t sizeof_detHit = bitset_size(sim->detHit) * sizeof(uint);

    //TO_HOST(sim->lenTiss, gmem->lenTiss, sizeof_tissueArrays);
    //TO_HOST(sim->momTiss, gmem->momTiss, sizeof_tissueArrays);
    TO_HOST(sim->II, gmem->II, sizeof_II);
    TO_HOST(sim->detHit.matrix, gmem->detHit.matrix, sizeof_detHit);
}
