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
#include <math.h>
#include <string.h>

#include "bitset2d.h"

#define PI M_PI
#define C_VACUUM 2.9979e11
#define FP_DIV_ERR 1e-8
#define EPS 2.2204e-16
#define MIN(a,b) ((a) < (b) ? (a) :  (b))
#define absf(x)  ((x) >  0  ? (x) : -(x))

#define TO_DEVICE(d_ptr, h_ptr, size) (cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice))
#define TO_HOST(d_ptr, h_ptr, size)   (cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyDeviceToHost))

#define LIN2D(x,y,max_x) ((x) + (y) * (max_x))
#define LIN3D(x,y,z,max_x,max_y) (LIN2D((x),(y),(max_x)) + (z) * ((max_x) * (max_y)))

#define DIST2VOX(x, sr) ((int) ((x) * (sr)))

typedef unsigned int uint;
typedef unsigned char uchar;
typedef float Real;

// TODO: find a better place for this?
#define RAND_BUF_LEN  5     // Register arrays
#define RAND_SEED_LEN 5     // 32bit seed length (32*5 = 160bits)

#define MAX_DETECTORS 256
#define MAX_TISSUES 128

typedef struct {
    unsigned char ***tissueType; // type of the tissue within the voxel
    int3 dim;       // dimensions of the image file
    float3 stepr;   // inverse of voxel dimensions
    Real minstepsize;

    // Apparently this restricts the photon fluence calculation to within
    // a box outlined by the following coordinates.
    int3 Imin, Imax;

    // TODO: find better names
    int3 nIstep;
    int nIxy, nIxyz;
} Grid;

typedef struct {
    float3 r;   // initial position of the photon (euclidean)
    float3 d;   // initial direction cosines 
} Source;

typedef struct {
    int num;        // specify number of detectors
    Real radius;    // specify detector radius 
    int **loc;      // and x,y,z locations 
} Detectors;

typedef struct {
    int num;

    // Optical properties of the different tissue types
    Real *musr, *mua;
    Real *g, *n;
} Tissue;

typedef struct {
    int n_photons;

    Real min_length, max_length;
    Real stepT, stepLr;
    Real max_time;
    Bitset2D detHit;

    Real *lenTiss, *momTiss;
    Real *II;

    Grid grid;
    Source src;
    Detectors det;
    Tissue tiss;
} Simulation;

// Structure holding pointers to the GPU global memory.
typedef struct {
    // Tissue type index of each voxel.
    unsigned char *tissueType;

    // Location (grid) of each detector.
    //int3 *detLoc;

    // Optical properties of the different tissue types.
    //Real *tmusr, *tmua;
    //Real *tg;

    // Path length and momentum transfer.
    Real *lenTiss, *momTiss;

    // Photon fluence
    Real *II;

    // Bitset of detectors hit by a given photon packet.
    Bitset2D detHit;

    // Seed for the random number generator.
    uint *seed;
} GPUMemory;

typedef struct {
    int n_blocks, n_threads, n_threads_per_block;
    int rand_seed;
} ExecConfig;

// Function prototypes
// TODO: Should we be consistent with the arguments, even though
//       it's not necessary? Investigate.
extern int read_input(ExecConfig *conf, Simulation *sim, const char *input_filename);
extern void init_mem(ExecConfig conf, Simulation *sim, GPUMemory *gmem);
extern void free_mem(Simulation sim, GPUMemory gmem); 
extern void retrieve(Simulation *sim, GPUMemory *gmem);
extern void write_results(Simulation sim, const char *input_filename);
extern void correct_source(Simulation *sim);
extern void simulate(ExecConfig conf, Simulation sim, GPUMemory gmem);

// Constant memory on the GPU
// TODO: profile, profile, profile.
//__constant__ Simulation s;
//__constant__ GPUMemory g;

#endif // _MAIN_H_
