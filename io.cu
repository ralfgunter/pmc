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
    char ***tissueType;
    int i,j,k;

    printf( "Loading target medium volume from %s\n", filename );

    // Read in the segmented data file
    fp = fopen( filename, "rb" );
    if( fp == NULL ) {
        printf( "ERROR: The binary image file %s was not found!\n", filename );
        return -1;    // TODO: better error handling (possibly through CUDA?)
    }

    tissueType = (char ***) malloc(sim->grid.dim_x * sizeof(char **));
    for( i = 0; i < sim->grid.dim_x; i++ ) {
        tissueType[i] = (char **) malloc(sim->grid.dim_y * sizeof(char *));
        for( j = 0; j < sim->grid.dim_y; j++ ) {
            tissueType[i][j] = (char *) malloc(sim->grid.dim_z * sizeof(char));
        }
    }

    for( k = 0; k < sim->grid.dim_z; k++ )
        for( j = 0; j < sim->grid.dim_y; j++ )
            for( i = 0; i < sim->grid.dim_x; i++ )
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

    int dim_x, dim_y, dim_z;    // dimensions of the image file
    double xstep, ystep, zstep; // voxel dimensions
    double minstepsize;

    Real *tmusr, *tmua;  // optical properties of the different tissue types
    Real *tg, *tn;

    double src_x, src_y, src_z;     // initial position of the photon (euclidean)
    double src_dx, src_dy, src_dz;  // initial direction cosines 

    int Ixmin, Ixmax, Iymin, Iymax, Izmin, Izmax;   // min and max x,y,z for 
                                                    // storing the 2-pt fluence
    int nIxstep, nIystep, nIzstep;

    double minT, maxT;      // min and max time for sampling the 2-pt fluence 
    double stepT, stepL;    // time step and corresponding length step for sampling 
                            // the 2-pt fluence 
    double stepT_r, stepT_too_small;   // stepT_r remainder gate width 

    double min_length, max_length;  // min and max length allowed for the photon to propagate
    double max_time_float;
    int max_time, max_time_int;

    int nDets;          // specify number of detectors
    float detRad;       // specify detector radius 
    int **detLoc;       // and x,y,z locations 

    FILE *fp;
    char segFile[32];   // file name for image file 

    fp = fopen( filename, "r" );
    if( fp == NULL ) {
        printf( "usage: tMCimg input_file\n" );
        printf( "input_file = %s does not exist.\n", filename );
        exit(1);    // TODO: better error handling (possibly through CUDA?)
    }

    // Read the input file 
    fscanf( fp, "%d", &n_photons );                         // total number of photons
    fscanf( fp, "%d", &rand_seed );                         // random number seed
    fscanf( fp, "%lf %lf %lf", &src_x, &src_y, &src_z );    // source location
    fscanf( fp, "%lf %lf %lf", &src_dx, &src_dy, &src_dz ); // initial direction of photon
    fscanf( fp, "%lf %lf %lf", &minT, &maxT, &stepT );      // min, max, step time for recording
    fscanf( fp, "%s", segFile );                            // file containing tissue structure

    // Read image dimensions
    fscanf( fp, "%lf %d %d %d", &xstep, &dim_x, &Ixmin, &Ixmax );
    fscanf( fp, "%lf %d %d %d", &ystep, &dim_y, &Iymin, &Iymax );
    fscanf( fp, "%lf %d %d %d", &zstep, &dim_z, &Izmin, &Izmax );
    Ixmin--; Ixmax--; Iymin--; Iymax--; Izmin--; Izmax--;
    nIxstep = Ixmax-Ixmin+1;
    nIystep = Iymax-Iymin+1;
    nIzstep = Izmax-Izmin+1;

    // Read number of tissue types and their optical properties
    fscanf( fp, "%d", &n_tissues );
    // Index 0 is used as a flag
    tmusr = (Real *) malloc((n_tissues + 1) * sizeof(Real));
    tmua  = (Real *) malloc((n_tissues + 1) * sizeof(Real));
    tg    = (Real *) malloc((n_tissues + 1) * sizeof(Real));
    tn    = (Real *) malloc((n_tissues + 1) * sizeof(Real));
    tmusr[0] = (1.0 / -999.0); tmua[0] = -999.0; tg[0] = -999.0; tn[0] = -999.0;
    for( i = 1; i <= n_tissues; i++ ) {
        float tmus;
        // TODO: allow Real = double as well
        fscanf( fp, "%f %f %f %f", &tmus, &tg[i], &tmua[i], &tn[i] );
        if( tn[i] != 1.0 ) {
            printf( "WARNING: The code does not yet support n != 1.0\n" );
            printf( "tn[%d] = %f\n", i, tn[i] );
        }
        if( tmus == 0.0 ) {
            printf( "ERROR: The code does not support mus = 0.0\n" );
            return -1;
        }
        // The scattering coefficient is always used in the denominator of a
        // division, which is more computationally expensive to do than
        // multiplication, hence why we store its inverse here.
        tmusr[i] = 1.0 / tmus;
    }

    // Read number of detectors, detector radius, and detector locations
    fscanf( fp, "%d %f", &nDets, &detRad );

    detLoc = (int **) malloc(nDets * sizeof(int *));
    for( i = 0; i < nDets; i++ ) {
        double det_x, det_y, det_z;
        detLoc[i] = (int *) malloc(3 * sizeof(int));

        fscanf( fp, "%lf %lf %lf", &det_x, &det_y, &det_z);
        detLoc[i][0] = (int) (det_x / xstep) - 1;
        detLoc[i][1] = (int) (det_y / ystep) - 1;
        detLoc[i][2] = (int) (det_z / zstep) - 1;
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
    minstepsize = MIN(xstep, MIN(ystep, zstep)); 

    // Normalize the direction cosine of the source
    Real foo = sqrt(src_dx*src_dx + src_dy*src_dy + src_dz*src_dz);
    src_dx /= foo;
    src_dy /= foo;
    src_dz /= foo;

    // Calculate the min and max photon length from the min and max propagation times
    max_length = maxT * C_VACUUM / tn[1];
    min_length = minT * C_VACUUM / tn[1];
    stepL = stepT * C_VACUUM / tn[1];

    // Copy data to the simulation struct
    sim->max_time = max_time;
    sim->min_length = min_length;
    sim->max_length = max_length;
    sim->stepLr = 1.0 / stepL;  // as with tmusr
    sim->stepT = stepT;
    sim->n_photons = n_photons;

    sim->grid.minstepsize = minstepsize;
    sim->grid.Ixmin = Ixmin; sim->grid.Ixmax = Ixmax;
    sim->grid.Iymin = Iymin; sim->grid.Iymax = Iymax;
    sim->grid.Izmin = Izmin; sim->grid.Izmax = Izmax;
    sim->grid.dim_x = dim_x;
    sim->grid.dim_y = dim_y;
    sim->grid.dim_z = dim_z;
    sim->grid.xstep = xstep;
    sim->grid.ystep = ystep;
    sim->grid.zstep = zstep;
    sim->grid.nIxstep = nIxstep;
    sim->grid.nIystep = nIystep;
    sim->grid.nIzstep = nIzstep;
    sim->grid.nIxy  = nIxstep * nIystep;
    sim->grid.nIxyz = nIzstep * sim->grid.nIxy;

    sim->src.r.x = src_x;
    sim->src.r.y = src_y;
    sim->src.r.z = src_z;
    sim->src.d.x = src_dx;
    sim->src.d.y = src_dy;
    sim->src.d.z = src_dz;

    sim->det.num    = nDets;
    sim->det.radius = detRad;
    sim->det.loc    = detLoc;

    sim->tiss.num  = n_tissues;
    sim->tiss.musr = tmusr;
    sim->tiss.mua  = tmua;
    sim->tiss.g = tg;
    sim->tiss.n = tn;

    conf->rand_seed = rand_seed;

    read_segmentation_file(sim, segFile);

    return 0;
}

// TODO: handle the remaining files to be written.
void write_results(Simulation sim, const char *input_filename)
{
    FILE *history, *fluence, *momentum, *pathlength;
    char filename[32];
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
                    printf("(i, photonIndex) = (%d, %d)\n", i, photonIndex);

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
