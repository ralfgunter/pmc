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

int read_segmentation_file(Simulation *sim, const char *filename)
{
    FILE *fp;
    int i, j, k;
    uint8_t ***media_type;

    //printf( "Loading target medium volume from %s\n", filename );

    // Read in the segmented data file
    fp = fopen( filename, "rb" );
    if( fp == NULL ) {
        printf( "ERROR: The binary image file %s was not found!\n", filename );
        return -1;    // TODO: better error handling (possibly through CUDA?)
    }

    media_type = (uint8_t ***) malloc(sim->grid.dim.x * sizeof(uint8_t **));
    for( i = 0; i < sim->grid.dim.x; i++ )
    {
        media_type[i] = (uint8_t **) malloc(sim->grid.dim.y * sizeof(uint8_t *));
        for( j = 0; j < sim->grid.dim.y; j++ )
        {
            media_type[i][j] = (uint8_t *) malloc(sim->grid.dim.z * sizeof(uint8_t));
        }
    }

    for( k = 0; k < sim->grid.dim.z; k++ )
        for( j = 0; j < sim->grid.dim.y; j++ )
            for( i = 0; i < sim->grid.dim.x; i++ )
                fscanf( fp, "%c", &media_type[i][j][k] );

    sim->grid.media_type = media_type;

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

    float4 *media_prop;  // optical properties of the different tissue types

    float4 src_pos; // initial position of the photon (euclidean)
    float4 src_dir; // initial direction cosines 

    int3 fbox_min, fbox_max;    // min and max x,y,z for storing the fluence
    int3 fbox_dim;        // dimensions of the the region above

    double minT, maxT;      // min and max time for sampling the photon fluence 
    double time_step, stepL;    // time step and corresponding length step for sampling 
                            // the photon fluence 
    double time_step_r, time_step_too_small;   // time_step_r remainder gate width 

    double min_length, max_length;  // min and max length allowed for the photon to propagate
    double num_time_steps_float;
    int num_time_steps, num_time_steps_int;

    int num_dets;   // specify number of detectors
    int4 *det;      // grid position of each detector, plus its radius

    FILE *fp;
    char segFile[128];   // file name for image file 


    fp = fopen( filename, "r" );
    if( fp == NULL )
    {
        printf( "input_file = %s does not exist.\n", filename );
        return -1;    // TODO: better error handling (possibly through CUDA?)
    }

    // Read the input file .
    fscanf( fp, "%d", &n_photons );     // total number of photons
    fscanf( fp, "%d", &rand_seed );     // random number seed
    fscanf( fp, "%f %f %f", &src_pos.x, &src_pos.y, &src_pos.z );    // source location
    fscanf( fp, "%f %f %f", &src_dir.x, &src_dir.y, &src_dir.z );    // source direction
    fscanf( fp, "%lf %lf %lf", &minT, &maxT, &time_step );  // min, max, step time for recording
    fscanf( fp, "%s", segFile );                        // file containing tissue structure

    // Read image dimensions.
    fscanf( fp, "%f %d %d %d", &vox_dim.x, &grid_dim.x, &fbox_min.x, &fbox_max.x );
    fscanf( fp, "%f %d %d %d", &vox_dim.y, &grid_dim.y, &fbox_min.y, &fbox_max.y );
    fscanf( fp, "%f %d %d %d", &vox_dim.z, &grid_dim.z, &fbox_min.z, &fbox_max.z );
    fbox_min.x--; fbox_max.x--; fbox_min.y--; fbox_max.y--; fbox_min.z--; fbox_max.z--;
    fbox_dim.x = fbox_max.x - fbox_min.x + 1;
    fbox_dim.y = fbox_max.y - fbox_min.y + 1;
    fbox_dim.z = fbox_max.z - fbox_min.z + 1;

    // Read number of tissue types and their optical properties.
    fscanf( fp, "%d", &n_tissues );
    // index 0 is used as a flag
    media_prop = (float4 *) malloc((n_tissues + 1) * sizeof(float4));
    media_prop[0].x = (1.0 / -999.0);
    media_prop[0].y = -999.0;
    media_prop[0].z = -999.0;
    media_prop[0].w = -999.0;
    for( i = 1; i <= n_tissues; i++ )
    {
        float tmus;
        fscanf( fp, "%f %f %f %f", &tmus, &media_prop[i].y, &media_prop[i].z, &media_prop[i].w );
        if( media_prop[i].w != 1.0 )
        {
            printf( "WARNING: The code does not yet support n != 1.0\n" );
        }
        if( tmus == 0.0 ) {
            printf( "ERROR: The code does not support mus = 0.0\n" );
            return -1;
        }
        // The scattering coefficient is always used in the denominator of a
        // division, which is more computationally expensive to do than
        // multiplication, hence why we store its inverse here.
        media_prop[i].x = 1.0 / tmus;
    }

    // Read number of detectors, their radius and locations.
    float radius;
    fscanf( fp, "%d %f", &num_dets, &radius);

    double det_x, det_y, det_z;
    det = (int4 *) malloc(num_dets * sizeof(int4));
    for( i = 0; i < num_dets; i++ )
    {
        fscanf( fp, "%lf %lf %lf", &det_x, &det_y, &det_z);
        det[i].x = (int) (det_x / vox_dim.x);
        det[i].y = (int) (det_y / vox_dim.y);
        det[i].z = (int) (det_z / vox_dim.z);
        det[i].w = (int) radius;
    }

    fclose(fp);

    // Calculate number of gates, taking into account floating point division errors.
    num_time_steps_float = (maxT - minT) / time_step;
    num_time_steps_int   = (int) num_time_steps_float;
    time_step_r = absf(num_time_steps_float - num_time_steps_int) * time_step;
    time_step_too_small = FP_DIV_ERR * time_step;
    if(time_step_r < time_step_too_small)
        num_time_steps = num_time_steps_int;
    else
        num_time_steps = ceil(num_time_steps_float);

    // Get the minimum dimension.
    minstepsize = MIN(vox_dim.x, MIN(vox_dim.y, vox_dim.z)); 

    // Normalize the direction cosine of the source.
    float foo = sqrt(src_dir.x*src_dir.x + src_dir.y*src_dir.y + src_dir.z*src_dir.z);
    src_dir.x /= foo;
    src_dir.y /= foo;
    src_dir.z /= foo;

    // Calculate the min/max photon trajectory length from the min/max propagation time.
    max_length = maxT * C_VACUUM / media_prop[1].w;
    min_length = minT * C_VACUUM / media_prop[1].w;
    stepL     = time_step * C_VACUUM / media_prop[1].w;

    // Copy data to the simulation struct.
    sim->num_time_steps = num_time_steps;
    sim->min_length = min_length;
    sim->max_length = max_length;
    sim->stepLr = 1.0 / stepL;  // as with tmus
    sim->time_step = time_step;
    sim->n_photons = n_photons;

    sim->grid.minstepsize = minstepsize;
    sim->grid.fbox_min = fbox_min; sim->grid.fbox_max = fbox_max;
    sim->grid.dim = grid_dim;
    sim->grid.stepr.x = 1.0 / vox_dim.x; // as with tmus
    sim->grid.stepr.y = 1.0 / vox_dim.y; // as with tmus
    sim->grid.stepr.z = 1.0 / vox_dim.z; // as with tmus
    sim->grid.fbox_dim = fbox_dim;
    sim->grid.nIxy  = fbox_dim.x * fbox_dim.y;
    sim->grid.nIxyz = fbox_dim.z * sim->grid.nIxy;

    sim->src.r = src_pos;
    sim->src.d = src_dir;

    sim->det.num  = num_dets;
    sim->det.info = det;

    sim->tiss.num = n_tissues;
    sim->tiss.prop = media_prop;

    conf->rand_seed = rand_seed;

    read_segmentation_file(sim, segFile);

    return 0;
}

int write_results(Simulation sim, const char *input_filename)
{
    FILE *history, *fluence, *dyn;//*momentum, *path_length;
    char filename[128];
    int8_t det_idx;
    int media_idx;
    uint32_t photon_idx, k;

    // TODO: check for errors
    sprintf( filename, "%s.his", input_filename );
    history = fopen( filename, "wb" );
    sprintf( filename, "%s.dyn", input_filename );
    dyn = fopen( filename, "w" );
    //momentum   = fopen( "momentum_transfer", "w" );
    //path_length = fopen( "path_length", "w" );

    if( sim.det.num != 0 )
    {
        for( photon_idx = 0; photon_idx < sim.n_photons; photon_idx++ )
        {
            if( (det_idx = sim.det.hit[photon_idx]) != 0 )
            {
                // Write to the history file
                fwrite(&(--det_idx), sizeof(int), 1, history);
                for( media_idx = 1; media_idx <= sim.tiss.num; media_idx++ )
                {
                    k = MAD_IDX(photon_idx, media_idx);

                    fwrite(&sim.path_length[k], sizeof(float), 1, history);
                    fprintf(dyn, "%f %f\n", sim.path_length[k], sim.mom_transfer[k]);
                    //fprintf(path_length, "%f\n", sim.path_length[k]);
                    //fprintf(momentum,   "%f\n", sim.mom_transfer[k]);       
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
            fwrite( &sim.path_length[j], sizeof(float), 1, history );
        }
        for( j = 1; j <= s.tiss.num; j++ ) {
            fwrite( &sim.mom_transfer[j], sizeof(float), 1, history );
        }
    }
*/

    // Save fluence data
    sprintf( filename, "%s.2pt", input_filename );
    fluence = fopen( filename, "wb" );
    if(fluence != NULL) {
        fwrite( sim.fbox, sizeof(float), sim.grid.nIxyz * sim.num_time_steps, fluence );
    } else {
        printf( "ERROR: unable to save to %s\n", filename );
        return -1;
    }

    // Close file handlers.
    fclose(history);
    fclose(fluence);
    fclose(dyn);
    //fclose(momentum);
    //fclose(path_length);

    return 0;
}

void parse_conf(ExecConfig *conf, int n_threads, int n_iterations)
{
    conf->n_threads = n_threads;
    conf->n_blocks = conf->n_threads / 128;
    conf->n_iterations = n_iterations;

    if(conf->rand_seed > 0)
        srand(conf->rand_seed);
    else
        srand(time(NULL));
}
