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

#ifndef _MAIN_H_
#define _MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PI 3.1415926535897932
#define C_VACUUM 2.9979e11
#define FP_DIV_ERR 1e-8
#define EPS 2.2204e-16

// TODO: find a better place for this?
#define RAND_BUF_LEN  5     // Register arrays
#define RAND_SEED_LEN 5     // 32bit seed length (32*5 = 160bits)

#define MAX_DETECTORS 256
#define MAX_TISSUES 128

#define MIN(a,b) ((a) < (b) ? (a) :  (b))
#define absf(x)  ((x) >  0  ? (x) : -(x))

#define TO_DEVICE(d_ptr, h_ptr, size) (cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice))
#define TO_HOST(d_ptr, h_ptr, size)   (cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyDeviceToHost))

#define LIN2D(x,y,max_x) ((x) + (y) * (max_x))
#define LIN3D(x,y,z,max_x,max_y) (LIN2D((x),(y),(max_x)) + (z) * ((max_x) * (max_y)))
#define LIN(i,j,k,time,grid) (time * grid.nIxyz + \
                              ((k) - grid.fbox_min.z) * grid.nIxy + \
                              ((j) - grid.fbox_min.y) * grid.fbox_dim.x + \
                              ((i) - grid.fbox_min.x))

// Magic number is "any odd number with a decent mix of 0s and 1s in every byte"
// - SPWorley at http://forums.nvidia.com/index.php?showtopic=189165
// multiply-add code from wikipedia
#define NUM_HASH_BITS 25
#define MAD_HASH(key) ((unsigned) (0x27253271 * (key)) >> (32 - NUM_HASH_BITS + 1))
#define MAD_IDX(x, y) MAD_HASH((x << 5) | y)

typedef struct {
    uint8_t ***media_type; // type of the tissue within the voxel
    int3 dim;       // dimensions of the image file
    float3 stepr;   // inverse of voxel dimensions
    float minstepsize;

    // This restricts the photon fluence calculation to within
    // a box outlined by the following coordinates.
    int3 fbox_min, fbox_max;

    // TODO: find better names
    int3 fbox_dim;
    int nIxy, nIxyz;
} Grid;

typedef struct {
    float4 r;   // initial position of the photon (euclidean)
    float4 d;   // initial direction cosines 
} Source;

typedef struct {
    int num;    // specify number of detectors
    int4 *info; // grid coordinates and and radius
    int8_t *hit;
} Detectors;

typedef struct {
    int num;
    float4 *prop; // Optical properties
} Tissue;

typedef struct {
    uint32_t n_photons;

    float min_length, max_length;
    float time_step, stepLr;
    int num_time_steps;

    float *path_length, *mom_transfer;
    float *fbox;

    Grid grid;
    Source src;
    Detectors det;
    Tissue tiss;
} Simulation;

// Structure holding pointers to the GPU global memory.
typedef struct {
    // Tissue type index of each voxel.
    uint8_t *media_type;

    // Location (grid) of each detector, plus its radius.
    int4 *det_loc;

    // Optical properties of the different tissue types.
    float4 *media_prop;

    // Path length and momentum transfer.
    float *path_length, *mom_transfer;

    // Photon fluence
    float *fbox;

    // Detector hit by a given photon.
    int8_t *det_hit;

    // Seed for the random number generator.
    uint32_t *seed;
} GPUMemory;

typedef struct {
    int n_blocks, n_threads;
    int n_iterations;
    int rand_seed;
} ExecConfig;

// Function prototypes
// TODO: Should we be consistent with the arguments, even though
//       it's not necessary? Investigate.
extern int read_input(ExecConfig *conf, Simulation *sim, const char *input_filename);
extern int write_results(Simulation sim, const char *input_filename);
extern int read_segmentation_file(Simulation *sim, const char *filename);
extern void init_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem);
extern void free_mem(Simulation sim, GPUMemory gmem); 
extern void free_cpu_params_mem(Simulation sim); 
extern void free_gpu_params_mem(GPUMemory gmem); 
extern void free_cpu_results_mem(Simulation sim); 
extern void free_gpu_results_mem(GPUMemory gmem); 
extern void retrieve(Simulation *sim, GPUMemory *gmem);
extern void correct_source(Simulation *sim);
extern void simulate(ExecConfig conf, Simulation sim, GPUMemory gmem);
extern void parse_conf(ExecConfig *conf, int n_threads, int n_iterations);
extern uint32_t* init_rand_seed(int seed, ExecConfig conf);

#endif // _MAIN_H_
