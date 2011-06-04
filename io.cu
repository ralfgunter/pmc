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

int read_segmentation_file(Simulation *sim, const char *filename)
{
    FILE *fp;
    uchar ***tissueType;
    int i,j,k;

    printf( "Loading target medium volume from %s\n", filename );

    // Read in the segmented data file
    fp = fopen( filename, "rb" );
    if( fp == NULL ) {
        printf( "ERROR: The binary image file %s was not found!\n", filename );
        return -1;    // TODO: better error handling (possibly through CUDA?)
    }

    tissueType = (uchar ***) malloc(sim->grid.dim.x * sizeof(uchar **));
    for( i = 0; i < sim->grid.dim.x; i++ ) {
        tissueType[i] = (uchar **) malloc(sim->grid.dim.y * sizeof(uchar *));
        for( j = 0; j < sim->grid.dim.y; j++ ) {
            tissueType[i][j] = (uchar *) malloc(sim->grid.dim.z * sizeof(uchar));
        }
    }

    for( k = 0; k < sim->grid.dim.z; k++ )
        for( j = 0; j < sim->grid.dim.y; j++ )
            for( i = 0; i < sim->grid.dim.x; i++ )
                fscanf( fp, "%c", &tissueType[i][j][k] );

    sim->grid.tissueType = tissueType;

    fclose(fp);

    return 0;
}

/*********************************************************
    OPEN AND READ THE INPUT FILE 
 *********************************************************/
int read_input(ExecConfig *conf, Simulation *sim, const char *filename)
{
    int i;          // loop index
    int rand_seed;  // seed for the RNG
    int n_photons;  // total number of photons to run
    int n_tissues;  // number of tissue types described in the image file

    int3 grid_dim;   // dimensions of the image file
    float3 vox_dim; // voxel dimensions
    double minstepsize;

    float4 *tissProp;  // optical properties of the different tissue types

    float4 src_pos; // initial position of the photon (euclidean)
    float4 src_dir; // initial direction cosines 

    int3 Imin, Imax;    // min and max x,y,z for storing the fluence
    int3 nIstep;        // dimensions of the the region above

    double minT, maxT;      // min and max time for sampling the photon fluence 
    double stepT, stepL;    // time step and corresponding length step for sampling 
                            // the photon fluence 
    double stepT_r, stepT_too_small;   // stepT_r remainder gate width 

    double min_length, max_length;  // min and max length allowed for the photon to propagate
    double max_time_float;
    int max_time, max_time_int;

    int num_dets;   // specify number of detectors
    int4 *det;      // grid position of each detector, plus its radius

    FILE *fp;
    char segFile[128];   // file name for image file 


    fp = fopen( filename, "r" );
    if( fp == NULL ) {
        printf( "usage: tMCimg input_file\n" );
        printf( "input_file = %s does not exist.\n", filename );
        exit(1);    // TODO: better error handling (possibly through CUDA?)
    }

    // Read the input file 
    fscanf( fp, "%d", &n_photons );                         // total number of photons
    fscanf( fp, "%d", &rand_seed );                         // random number seed
    fscanf( fp, "%f %f %f", &src_pos.x, &src_pos.y, &src_pos.z );    // source location
    fscanf( fp, "%f %f %f", &src_dir.x, &src_dir.y, &src_dir.z );    // source src_direction
    fscanf( fp, "%lf %lf %lf", &minT, &maxT, &stepT );      // min, max, step time for recording
    fscanf( fp, "%s", segFile );                            // file containing tissue structure

    // Read image dimensions
    fscanf( fp, "%f %d %d %d", &vox_dim.x, &grid_dim.x, &Imin.x, &Imax.x );
    fscanf( fp, "%f %d %d %d", &vox_dim.y, &grid_dim.y, &Imin.y, &Imax.y );
    fscanf( fp, "%f %d %d %d", &vox_dim.z, &grid_dim.z, &Imin.z, &Imax.z );
    Imin.x--; Imax.x--; Imin.y--; Imax.y--; Imin.z--; Imax.z--;
    nIstep.x = Imax.x - Imin.x + 1;
    nIstep.y = Imax.y - Imin.y + 1;
    nIstep.z = Imax.z - Imin.z + 1;

    // Read number of tissue types and their optical properties
    fscanf( fp, "%d", &n_tissues );
    // Index 0 is used as a flag
    tissProp = (float4 *) malloc((n_tissues + 1) * sizeof(float4));
    tissProp[0].x = (1.0 / -999.0);
    tissProp[0].y = -999.0;
    tissProp[0].z = -999.0;
    tissProp[0].w = -999.0;
    for( i = 1; i <= n_tissues; i++ ) {
        float tmus;
        // TODO: allow Real = double as well
        fscanf( fp, "%f %f %f %f", &tmus, &tissProp[i].y, &tissProp[i].z, &tissProp[i].w );
        if( tissProp[i].w != 1.0 ) {
            printf( "WARNING: The code does not yet support n != 1.0\n" );
            printf( "tn[%d] = %f\n", i, tissProp[i].w );
        }
        if( tmus == 0.0 ) {
            printf( "ERROR: The code does not support mus = 0.0\n" );
            return -1;
        }
        // The scattering coefficient is always used in the denominator of a
        // division, which is more computationally expensive to do than
        // multiplication, hence why we store its inverse here.
        tissProp[i].x = 1.0 / tmus;
    }

    // Read number of detectors, their radius and locations.
    float radius;
    fscanf( fp, "%d %f", &num_dets, &radius);

    det = (int4 *) malloc(num_dets * sizeof(int4));
    for( i = 0; i < num_dets; i++ ) {
        double det_x, det_y, det_z;

        fscanf( fp, "%lf %lf %lf", &det_x, &det_y, &det_z);
        det[i].x = (int) (det_x / vox_dim.x) - 1;
        det[i].y = (int) (det_y / vox_dim.y) - 1;
        det[i].z = (int) (det_z / vox_dim.z) - 1;
        det[i].w = (int) radius;
    }

    fclose(fp);

    // Calculate number of gates, taking into account floating point division errors.
    max_time_float = (maxT - minT) / stepT;
    max_time_int   = (int) max_time_float;
    stepT_r = absf(max_time_float - max_time_int) * stepT;
    stepT_too_small = FP_DIV_ERR * stepT;
    if(stepT_r < stepT_too_small)
        max_time = max_time_int;
    else
        max_time = ceil(max_time_float);

    // Get the minimum dimension
    minstepsize = MIN(vox_dim.x, MIN(vox_dim.y, vox_dim.z)); 

    // Normalize the direction cosine of the source
    Real foo = sqrt(src_dir.x*src_dir.x + src_dir.y*src_dir.y + src_dir.z*src_dir.z);
    src_dir.x /= foo;
    src_dir.y /= foo;
    src_dir.z /= foo;

    // Calculate the min and max photon length from the min and max propagation times
    max_length = maxT * C_VACUUM / tissProp[1].w;
    min_length = minT * C_VACUUM / tissProp[1].w;
    stepL = stepT * C_VACUUM / tissProp[1].w;

    // Copy data to the simulation struct
    sim->max_time = max_time;
    sim->min_length = min_length;
    sim->max_length = max_length;
    sim->stepLr = 1.0 / stepL;  // as with tmusr
    sim->stepT = stepT;
    sim->n_photons = n_photons;

    sim->grid.minstepsize = minstepsize;
    sim->grid.Imin = Imin; sim->grid.Imax = Imax;
    sim->grid.dim = grid_dim;
    sim->grid.stepr.x = 1.0 / vox_dim.x; // as with tmusr
    sim->grid.stepr.y = 1.0 / vox_dim.y; // as with tmusr
    sim->grid.stepr.z = 1.0 / vox_dim.z; // as with tmusr
    sim->grid.nIstep = nIstep;
    sim->grid.nIxy  = nIstep.x * nIstep.y;
    sim->grid.nIxyz = nIstep.z * sim->grid.nIxy;

    sim->src.r = src_pos;
    sim->src.d = src_dir;

    sim->det.num  = num_dets;
    sim->det.info = det;

    sim->tiss.num = n_tissues;
    sim->tiss.prop = tissProp;

    conf->rand_seed = rand_seed;

    read_segmentation_file(sim, segFile);

    return 0;
}

// TODO: handle the remaining files to be written.
void write_results(Simulation sim, const char *input_filename)
{
    FILE *history, *fluence, *momentum, *pathlength;
    char filename[128];
    int i, j, k, photonIndex;

    // TODO: check for errors
    sprintf( filename, "%s.his", input_filename );
    history    = fopen( filename, "wb" );
    momentum   = fopen( "momentum_transfer", "wb" );
    pathlength = fopen( "pathlength", "wb" );

    if( sim.det.num != 0 )
    {
        for(photonIndex = 0; photonIndex < sim.n_photons; photonIndex++)
        {
            // Loop through number of detectors
            for( i = 0; i < sim.det.num; i++ )
            {
                if(bitset_get(sim.detHit, photonIndex, i) == 1)
                {
                    // Write to the history file
                    fwrite(&i, sizeof(int), 1, history);
                    for( j = 1; j <= sim.tiss.num; j++ )
                    {
                        k = LIN2D(photonIndex, j, sim.n_photons);
                        fwrite(&sim.lenTiss[k], sizeof(float), 1, history);
                        fprintf(pathlength, "%f\n", sim.lenTiss[k]);
                    }
                    for( j = 1; j <= sim.tiss.num; j++ )
                    {
                        k = LIN2D(photonIndex, j, sim.n_photons);
                        fprintf(momentum, "%f\n", sim.momTiss[k]);       
                    }
                }
            }
        }
    }

/*    
    // If there are no detectors, then save exit position.
    } else {
        fwrite( &p.x, sizeof(float), 1, history );
        fwrite( &p.y, sizeof(float), 1, history );
        fwrite( &p.z, sizeof(float), 1, history );
        for( j = 1; j <= s.tiss.num; j++ ) {
            fwrite( &sim.lenTiss[j], sizeof(float), 1, history );
        }
        for( j = 1; j <= s.tiss.num; j++ ) {
            fwrite( &sim.momTiss[j], sizeof(float), 1, history );
        }
    }
*/

    // Save fluence data
    sprintf( filename, "%s.2pt", input_filename );
    fluence = fopen( filename, "wb" );
    if(fluence != NULL) {
        fwrite( sim.II, sizeof(Real), sim.grid.nIxyz * sim.max_time, fluence );
    } else {
        printf( "ERROR: unable to save to %s\n", filename );
        exit(1);
    }

    // Close file handlers.
    fclose(history);
    fclose(fluence);
    fclose(momentum);
    fclose(pathlength);
}
