#include <Python.h>
#include "structmember.h"
#include "main.h"
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <numpy/arrayobject.h>

#if PYTHON == 2
#undef PyLong_Check
#undef PyLong_AsLong 
#undef PyLong_FromLong 

#define PyLong_Check PyInt_Check
#define PyLong_AsLong PyInt_AsLong
#define PyLong_FromLong PyInt_FromLong
#endif

typedef struct {
    PyObject_HEAD

    PyObject *py_pathlength, *py_momentum_transfer;
    PyObject *py_medium;
    PyArrayObject *py_fluence;

    ExecConfig conf;
    Simulation sim;
    GPUMemory gmem;
} PyPMC;

static PyObject* pypmc_get_tissueArray( Simulation sim, float *tissueArray );
static PyArrayObject* pypmc_fluence_to_ndarray( Simulation sim, float *II );

////////////////////////////////////////////////////////////////////
//// Fundamental methods
static void
pypmc_dealloc( PyPMC *self )
{
    // Call PMC's own cleaning up procedure.
    free_mem(self->sim, self->gmem);

    Py_XDECREF(self->py_pathlength);
    Py_XDECREF(self->py_momentum_transfer);

    // Finally, delete the pypmc object itself.
#if PYTHON == 3
    Py_TYPE(self)->tp_free((PyObject *) self);
#elif PYTHON == 2
    self->ob_type->tp_free((PyObject *) self);
#endif
}

static PyObject *
pypmc_new( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
    PyPMC *self = (PyPMC *) type->tp_alloc(type, 0);

    return (PyObject *) self;
}

static int
pypmc_init( PyPMC *self, PyObject *args )
{
    const char *input_filepath = NULL;
    int n_threads, n_iterations;

    // The user may optionally pass the .inp filepath to automatically load
    // the simulation parameters from there.
    if(! PyArg_ParseTuple(args, "|sii", &input_filepath, &n_threads, &n_iterations))
        return -1;

    if(input_filepath == NULL)
        return 0;

    // Parse .inp file into the simulation structure.
    read_input(&self->conf, &self->sim, input_filepath);

    parse_conf(&self->conf, n_threads, n_iterations);

    // Make sure the source is at an interface.
    correct_source(&self->sim);

    // Allocate and initialize memory to be used by the GPU.
    init_mem(self->conf, &self->sim, &self->gmem);

    return 0;
}

//// end of fundamental methods
//////////////////////////////////////////////////////////////////////////////

static PyObject *
pypmc_write_to_disk( PyPMC *self, PyObject *args )
{
    const char *output_filepath;    

    if (! PyArg_ParseTuple(args, "s", &output_filepath))
        return NULL;

    write_results(self->sim, output_filepath);

    Py_RETURN_NONE;
}

static PyObject *
pypmc_run( PyPMC *self, PyObject *args )
{
    // Run simulations on the GPU.
    simulate(self->conf, self->sim, self->gmem);

    Py_RETURN_NONE;
}

static PyObject *
pypmc_pull_results( PyPMC *self, PyObject *args )
{
    // Retrieve results to host.
    retrieve(&self->sim, &self->gmem);

    self->py_pathlength = pypmc_get_tissueArray(self->sim, self->sim.lenTiss);
    self->py_momentum_transfer = pypmc_get_tissueArray(self->sim, self->sim.momTiss);
    self->py_fluence = pypmc_fluence_to_ndarray(self->sim, self->sim.II);

    Py_RETURN_NONE;
}

static PyObject *
pypmc_push_parameters( PyPMC *self, PyObject *args )
{
    // Allocate and initialize memory to be used by the GPU.
    free_gpu_mem(self->gmem);
    init_mem(self->conf, &self->sim, &self->gmem);

    Py_RETURN_NONE;
}

static PyObject *
pypmc_load_medium( PyPMC *self, PyObject *args )
{
    const char *medium_filepath;
    int dim_x, dim_y, dim_z;

    if (! PyArg_ParseTuple(args, "siii", &medium_filepath, &dim_x, &dim_y, &dim_z))
        return NULL;

    self->sim.grid.dim.x = dim_x;
    self->sim.grid.dim.y = dim_y;
    self->sim.grid.dim.z = dim_z;

    read_segmentation_file(&self->sim, medium_filepath);

    // TODO: better handle this
    self->sim.grid.nIstep.x = dim_x;
    self->sim.grid.nIstep.y = dim_y;
    self->sim.grid.nIstep.z = dim_z;
    self->sim.grid.nIxy  = dim_x * dim_y;
    self->sim.grid.nIxyz = dim_x * dim_y * dim_z;

    Py_RETURN_NONE;
}

// TODO: investigate if this has undefined behavior
//       (using local data - ndarray - outside the function).
static PyArrayObject *
pypmc_fluence_to_ndarray( Simulation sim, float *c_fluence )
{
    PyArrayObject *ndarray;
    float *c_array;
    npy_intp dim[3];

    dim[0] = sim.grid.nIstep.x;
    dim[1] = sim.grid.nIstep.y;
    dim[2] = sim.grid.nIstep.z;

    ndarray = (PyArrayObject *) PyArray_SimpleNew(3, dim, NPY_FLOAT);
    c_array = (float *) PyArray_DATA(ndarray);

    for (int x = 0; x < dim[0]; x++)
        for (int y = 0; y < dim[1]; y++)
            for (int z = 0; z < dim[2]; z++)
                c_array[LIN(x,y,z,0,sim.grid)] = c_fluence[LIN(x,y,z,0,sim.grid)];

    return ndarray;
}

////////////////////////////////////////////////////////////////////
//// Getters and setters

//// Setters
// ExecConfig
static int
pypmc_set_n_threads( PyPMC *self, PyObject *value, void *closure )
{
    // TODO: make a macro out of this
    if(! PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The n_threads attribute must be an int");
        return -1;
    }

    self->conf.n_threads = PyLong_AsLong(value);
    self->conf.n_blocks = self->conf.n_threads / 128;

    return 0;
}

static int
pypmc_set_n_iterations( PyPMC *self, PyObject *value, void *closure )
{
    // TODO: make a macro out of this
    if(! PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The n_iterations attribute must be an int");
        return -1;
    }

    self->conf.n_iterations = PyLong_AsLong(value);

    return 0;
}

static int
pypmc_set_rand_seed( PyPMC *self, PyObject *value, void *closure )
{
    // TODO: make a macro out of this
    if(! PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The rand_seed attribute must be an int");
        return -1;
    }

    self->conf.rand_seed = PyLong_AsLong(value);

    return 0;
}

// Simulation
static int
pypmc_set_n_photons( PyPMC *self, PyObject *value, void *closure )
{
    // TODO: make a macro out of this
    if(! PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "The n_photons attribute must be an int");
        return -1;
    }

    self->sim.n_photons = PyLong_AsLong(value);

    return 0;
}

static int
pypmc_set_src_pos( PyPMC *self, PyObject *coords, void *closure )
{
    if (! (PyTuple_Check(coords) && PyTuple_Size(coords) == 3))
    {
        PyErr_SetString(PyExc_TypeError,
                        "The attribute must be a tuple with three elements");
        return -1;
    }

    self->sim.src.r.x = (float) PyFloat_AsDouble(PyTuple_GetItem(coords, 0));
    self->sim.src.r.y = (float) PyFloat_AsDouble(PyTuple_GetItem(coords, 1));
    self->sim.src.r.z = (float) PyFloat_AsDouble(PyTuple_GetItem(coords, 2));

    correct_source(&self->sim);

    return 0;
}

static int
pypmc_set_src_dir( PyPMC *self, PyObject *dir_cosines, void *closure )
{
    float3 src_dir;

    if (! (PyTuple_Check(dir_cosines) && PyTuple_Size(dir_cosines) == 3))
    {
        PyErr_SetString(PyExc_TypeError,
                        "The source direction must be a tuple with three elements");
        return -1;
    }

    src_dir.x = (float) PyFloat_AsDouble(PyTuple_GetItem(dir_cosines, 0));
    src_dir.y = (float) PyFloat_AsDouble(PyTuple_GetItem(dir_cosines, 1));
    src_dir.z = (float) PyFloat_AsDouble(PyTuple_GetItem(dir_cosines, 2));

    // Normalize the direction cosine of the source.
    float foo = sqrt(src_dir.x*src_dir.x + src_dir.y*src_dir.y + src_dir.z*src_dir.z);
    src_dir.x /= foo;
    src_dir.y /= foo;
    src_dir.z /= foo;

    self->sim.src.d.x = src_dir.x;
    self->sim.src.d.y = src_dir.y;
    self->sim.src.d.z = src_dir.z;

    return 0;
}

static int
pypmc_set_detectors( PyPMC *self, PyObject *det_list, void *closure )
{
    PyObject *entry, *det_pos, *det_radius;
    Py_ssize_t num_detectors;

    if (! PyList_Check(det_list))
    {
        PyErr_SetString(PyExc_TypeError, "The detectors attribute must be a list");
        return -1;
    }

    // Each entry in the list uniquely identifies a detector.
    num_detectors = PyList_Size(det_list);
    self->sim.det.num = num_detectors;

    // The old detector list must be freed, and a new one built in its place.
    free(self->sim.det.info);
    self->sim.det.info = (int4 *) malloc(num_detectors * sizeof(int4));

    for (int i = 0; i < num_detectors; ++i)
    {
        entry = PyList_GetItem(det_list, i);
        det_pos = PyList_GetItem(entry, 0);
        det_radius = PyList_GetItem(entry, 1);

        self->sim.det.info[i].x = PyLong_AsLong(PyTuple_GetItem(det_pos, 0));
        self->sim.det.info[i].y = PyLong_AsLong(PyTuple_GetItem(det_pos, 1));
        self->sim.det.info[i].z = PyLong_AsLong(PyTuple_GetItem(det_pos, 2));
        self->sim.det.info[i].w = PyLong_AsLong(det_radius);
    }

    return 0;
}

static int
pypmc_set_tissues( PyPMC *self, PyObject *tissue_list, void *closure )
{
    PyObject *entry;
    Py_ssize_t num_tissues;

    if (! PyList_Check(tissue_list))
    {
        PyErr_SetString(PyExc_TypeError, "The tissues attribute must be a list");
        return -1;
    }

    // Each entry in the list uniquely identifies a tissue type.
    num_tissues = PyList_Size(tissue_list);
    self->sim.tiss.num = num_tissues;

    // The old tissue list must be freed, and a new one built in its place.
    free(self->sim.tiss.prop);
    self->sim.tiss.prop = (float4 *) malloc((num_tissues + 1) * sizeof(float4));

    for (int i = 0; i < num_tissues; ++i)
    {
        entry = PyList_GetItem(tissue_list, i);

        self->sim.tiss.prop[i + 1].x = (float) PyFloat_AsDouble(PyTuple_GetItem(entry, 0));
        self->sim.tiss.prop[i + 1].y = (float) PyFloat_AsDouble(PyTuple_GetItem(entry, 1));
        self->sim.tiss.prop[i + 1].z = (float) PyFloat_AsDouble(PyTuple_GetItem(entry, 2));
        self->sim.tiss.prop[i + 1].w = (float) PyFloat_AsDouble(PyTuple_GetItem(entry, 3));
    }

    return 0;
}

static int
pypmc_set_grid_dimensions( PyPMC *self, PyObject *dimensions, void *closure )
{
    PyObject *dim;

    // TODO: verify that every element is a tuple of two elements
    if (! (PyTuple_Check(dimensions) && PyTuple_Size(dimensions) == 3))
    {
        PyErr_SetString(PyExc_TypeError,
                        "The attribute must be a tuple with three elements");
        return -1;
    }

    dim = PyTuple_GetItem(dimensions, 0);
    self->sim.grid.dim.x = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.stepr.x = (float) PyFloat_AsDouble(PyTuple_GetItem(dim, 1));

    dim = PyTuple_GetItem(dimensions, 1);
    self->sim.grid.dim.y = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.stepr.y = (float) PyFloat_AsDouble(PyTuple_GetItem(dim, 1));

    dim = PyTuple_GetItem(dimensions, 2);
    self->sim.grid.dim.z = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.stepr.z = (float) PyFloat_AsDouble(PyTuple_GetItem(dim, 1));

    // Get the minimum dimension.
    self->sim.grid.minstepsize = MIN(1.0 / self->sim.grid.stepr.x,
                                     MIN(1.0 / self->sim.grid.stepr.y,
                                         1.0 / self->sim.grid.stepr.z)); 

    return 0;
}

static int
pypmc_set_fluence_box( PyPMC *self, PyObject *dimensions, void *closure )
{
    PyObject *dim;

    // TODO: verify that every element is a tuple of two elements
    if (! (PyTuple_Check(dimensions) && PyTuple_Size(dimensions) == 3))
    {
        PyErr_SetString(PyExc_TypeError,
                        "The attribute must be a tuple with three elements");
        return -1;
    }

    dim = PyTuple_GetItem(dimensions, 0);
    self->sim.grid.Imin.x = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.Imax.x = PyLong_AsLong(PyTuple_GetItem(dim, 1));
    self->sim.grid.nIstep.x = self->sim.grid.Imax.x - self->sim.grid.Imin.x + 1;

    dim = PyTuple_GetItem(dimensions, 1);
    self->sim.grid.Imin.y = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.Imax.y = PyLong_AsLong(PyTuple_GetItem(dim, 1));
    self->sim.grid.nIstep.y = self->sim.grid.Imax.y - self->sim.grid.Imin.y + 1;

    dim = PyTuple_GetItem(dimensions, 2);
    self->sim.grid.Imin.z = PyLong_AsLong(PyTuple_GetItem(dim, 0));
    self->sim.grid.Imax.z = PyLong_AsLong(PyTuple_GetItem(dim, 1));
    self->sim.grid.nIstep.z = self->sim.grid.Imax.z - self->sim.grid.Imin.z + 1;

    self->sim.grid.nIxy  = self->sim.grid.nIstep.x * self->sim.grid.nIstep.y;
    self->sim.grid.nIxyz = self->sim.grid.nIxy * self->sim.grid.nIstep.z;

    return 0;
}

static int
pypmc_set_time_params( PyPMC *self, PyObject *value, void *closure )
{
    double max_time, min_time, time_step;
    double max_time_float, stepT_r, stepT_too_small;
    int max_time_int;

    min_time  = PyFloat_AsDouble(PyTuple_GetItem(value, 0));
    max_time  = PyFloat_AsDouble(PyTuple_GetItem(value, 1));
    time_step = PyFloat_AsDouble(PyTuple_GetItem(value, 2));


    // Calculate number of gates, taking into account floating point division errors.
    max_time_float = (max_time - min_time) / time_step;
    max_time_int   = (int) max_time_float;
    stepT_r = absf(max_time_float - max_time_int) * time_step;
    stepT_too_small = FP_DIV_ERR * time_step;
    if(stepT_r < stepT_too_small)
        max_time = max_time_int;
    else
        max_time = ceil(max_time_float);

    // Calculate the min/max photon trajectory length from the min/max propagation time.
    self->sim.max_length     = max_time * C_VACUUM / self->sim.tiss.prop[1].w;
    self->sim.min_length     = min_time * C_VACUUM / self->sim.tiss.prop[1].w;
    self->sim.stepLr = 1.0 / (time_step * C_VACUUM / self->sim.tiss.prop[1].w);

    self->sim.max_time = max_time;
    self->sim.stepT = time_step;

    return 0;
}

//// Getters
// ExecConfig
static PyObject*
pypmc_get_n_threads( PyPMC *self, void *closure )
{
    return PyLong_FromLong(self->conf.n_threads);
}

static PyObject*
pypmc_get_n_iterations( PyPMC *self, void *closure )
{
    return PyLong_FromLong(self->conf.n_iterations);
}

static PyObject*
pypmc_get_rand_seed( PyPMC *self, void *closure )
{
    return PyLong_FromLong(self->conf.rand_seed);
}

// Simulation
static PyObject*
pypmc_get_n_photons( PyPMC *self, void *closure )
{
    return PyLong_FromLong(self->sim.n_photons);
}

static PyObject*
pypmc_get_src_pos( PyPMC *self, void *closure )
{
    PyObject *coords = Py_BuildValue("(fff)", self->sim.src.r.x,
                                              self->sim.src.r.y,
                                              self->sim.src.r.z);

    return coords;
}

static PyObject*
pypmc_get_src_dir( PyPMC *self, void *closure )
{
    PyObject *direction = Py_BuildValue("(fff)", self->sim.src.d.x,
                                                 self->sim.src.d.y,
                                                 self->sim.src.d.z);

    return direction;
}

static PyObject*
pypmc_get_detectors( PyPMC *self, void *closure )
{
    PyObject *det_list = Py_BuildValue("[]");
    PyObject *det_entry, *det_pos, *det_radius;

    for (int i = 0; i < self->sim.det.num; ++i)
    {
        det_pos = Py_BuildValue("(iii)", self->sim.det.info[i].x,
                                         self->sim.det.info[i].y,
                                         self->sim.det.info[i].z);
        det_radius = PyLong_FromLong(self->sim.det.info[i].w);
        det_entry = Py_BuildValue("[NN]", det_pos, det_radius);

        PyList_Append(det_list, det_entry);
    }

    return det_list;
}

static PyObject*
pypmc_get_tissues( PyPMC *self, void *closure )
{
    PyObject *entry;
    PyObject *tissue_list = Py_BuildValue("[]");

    for (int i = 1; i <= self->sim.tiss.num; ++i)
    {
        entry = Py_BuildValue("(ffff)", self->sim.tiss.prop[i].x,
                                        self->sim.tiss.prop[i].y,
                                        self->sim.tiss.prop[i].z,
                                        self->sim.tiss.prop[i].w);

        PyList_Append(tissue_list, entry);
    }

    return tissue_list;
}

static PyObject*
pypmc_get_grid_dimensions( PyPMC *self, void *closure )
{
    PyObject *dim_x, *dim_y, *dim_z;
    PyObject *dimensions;

    dim_x = Py_BuildValue("(if)", self->sim.grid.dim.x, self->sim.grid.stepr.x);
    dim_y = Py_BuildValue("(if)", self->sim.grid.dim.y, self->sim.grid.stepr.y);
    dim_z = Py_BuildValue("(if)", self->sim.grid.dim.z, self->sim.grid.stepr.z);

    dimensions = Py_BuildValue("(NNN)", dim_x, dim_y, dim_z);

    return dimensions;
}

static PyObject*
pypmc_get_fluence_box( PyPMC *self, void *closure )
{
    PyObject *dim_x, *dim_y, *dim_z;
    PyObject *dimensions;

    dim_x = Py_BuildValue("(ii)", self->sim.grid.Imin.x, self->sim.grid.Imax.x);
    dim_y = Py_BuildValue("(ii)", self->sim.grid.Imin.y, self->sim.grid.Imax.y);
    dim_z = Py_BuildValue("(ii)", self->sim.grid.Imin.z, self->sim.grid.Imax.z);

    dimensions = Py_BuildValue("(NNN)", dim_x, dim_y, dim_z);

    return dimensions;
}

static PyObject*
pypmc_get_tissueArray( Simulation sim, float *tissueArray )
{
    uint photonIndex, k;
    int detIndex, tissueIndex;
    PyObject *py_tissueArray = Py_BuildValue("[]");

    if( sim.det.num != 0 )
    {
        for( photonIndex = 0; photonIndex < sim.n_photons; photonIndex++ )
        {
            for( detIndex = 0; detIndex < sim.det.num; detIndex++ )
            {
                if( bitset_get(sim.detHit, photonIndex, detIndex) == 1 )
                {
                    for( tissueIndex = 1; tissueIndex <= sim.tiss.num; tissueIndex++ )
                    {
                        k = MAD_HASH((photonIndex << 5) | tissueIndex);

                        PyList_Append(py_tissueArray, PyFloat_FromDouble(tissueArray[k]));
                    }
                }
            }
        }
    }

    return py_tissueArray;
}

// FIXME: stub getter
static PyObject*
pypmc_get_time_params( PyPMC *self, void *closure )
{
    Py_RETURN_NONE;
}

//// end of getters and setters
////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////
//// Miscellaneous/Extension-related
static PyGetSetDef pypmc_getsetters[] = {
    // Execution parameters (ExecConfig) 
    {"n_threads",
     (getter) pypmc_get_n_threads, (setter) pypmc_set_n_threads, 
     "number of CUDA threads used in the simulation", NULL},
    {"n_iterations",
     (getter) pypmc_get_n_iterations, (setter) pypmc_set_n_iterations, 
     "number of CUDA iterations used in the simulation", NULL},
    {"rand_seed",
     (getter) pypmc_get_rand_seed, (setter) pypmc_set_rand_seed, 
     "number of CUDA iterations used in the simulation", NULL},

    // Simulation parameters (Simulation)
    {"n_photons",
     (getter) pypmc_get_n_photons, (setter) pypmc_set_n_photons, 
     "number of photons simulated", NULL},
    {"src_pos",
     (getter) pypmc_get_src_pos, (setter) pypmc_set_src_pos, 
     "euclidean position of the source (automatically corrected to be at an interface)", NULL},
    {"src_dir",
     (getter) pypmc_get_src_dir, (setter) pypmc_set_src_dir, 
     "direction cosines of the source", NULL},
    {"detectors",
     (getter) pypmc_get_detectors, (setter) pypmc_set_detectors, 
     "list of detectors (their position and radius)", NULL},
    {"tissues",
     (getter) pypmc_get_tissues, (setter) pypmc_set_tissues, 
     "list of tissues (their optical properties)", NULL},
    {"grid_dimensions",
     (getter) pypmc_get_grid_dimensions, (setter) pypmc_set_grid_dimensions, 
     "the grid's dimensions (voxel size and number of voxels in each direction)", NULL},
    {"fluence_box",
     (getter) pypmc_get_fluence_box, (setter) pypmc_set_fluence_box, 
     "the fluence box's vertices (in each direction)", NULL},
    {"time_params",
     (getter) pypmc_get_time_params, (setter) pypmc_set_time_params, 
     "(minimum time, maximum time, time step)", NULL},

    {NULL} /* Sentinel */
};

static PyMemberDef pypmc_members[] = {
    // Simulation results
    {"fluence", T_OBJECT_EX, offsetof(PyPMC, py_fluence), READONLY,
     "fluence"},
    {"pathlength", T_OBJECT_EX, offsetof(PyPMC, py_pathlength), READONLY,
     "the distance travelled by each photon in each type of tissue"},
    {"momentum_transfer", T_OBJECT_EX, offsetof(PyPMC, py_momentum_transfer), READONLY,
     "momentum transfer"},

    {NULL} /* Sentinel */
};

static PyMethodDef pypmc_methods[] = {
    {"run_simulation", (PyCFunction) pypmc_run, METH_NOARGS,
     "Does what it says on the tin."},
    {"pull_results", (PyCFunction) pypmc_pull_results, METH_NOARGS,
     "Transfers the simulation results to the host memory."},
    {"push_parameters", (PyCFunction) pypmc_push_parameters, METH_NOARGS,
     "Transfers the simulation parameters to the gpu memory."},
    {"write_to_disk", (PyCFunction) pypmc_write_to_disk, METH_VARARGS,
     "Saves the simulation results to disk, the old-fashioned way."},
    {"load_medium", (PyCFunction) pypmc_load_medium, METH_VARARGS,
     "Load tridimensional medium"},

    {NULL}  /* Sentinel */
};

static PyTypeObject pypmc_Type = {
#if PYTHON == 3
    PyVarObject_HEAD_INIT(NULL, 0)
#elif PYTHON == 2
    PyObject_HEAD_INIT(NULL)
    0,                  
#endif
    "pypmc.pypmc",                              /* tp_name */
    sizeof(PyPMC),                              /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor) pypmc_dealloc,                 /* tp_dealloc */
    0,                                          /* tp_print */
    0,                                          /* tp_getattr */
    0,                                          /* tp_setattr */
    0,                                          /* tp_compare */
    0,                                          /* tp_repr */
    0,                                          /* tp_as_number */
    0,                                          /* tp_as_sequence */
    0,                                          /* tp_as_mapping */
    0,                                          /* tp_hash */
    0,                                          /* tp_call */
    0,                                          /* tp_str */
    0,                                          /* tp_getattro */
    0,                                          /* tp_setattro */
    0,                                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "pypmc objects",                            /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    pypmc_methods,                              /* tp_methods */
    pypmc_members,                              /* tp_members */
    pypmc_getsetters,                           /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc) pypmc_init,                      /* tp_init */
    0,                                          /* tp_alloc */
    pypmc_new,                                  /* tp_new */
};

#if PYTHON == 3
static PyModuleDef pypmc_module = {
    PyModuleDef_HEAD_INIT,
    "pypmc",
    "Python bindings for PMC.",
    -1,
    NULL, NULL, NULL, NULL, NULL
};
#endif

#if PYTHON == 2
static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};
#endif

PyMODINIT_FUNC
#if PYTHON == 3
PyInit_pypmc(void)
{
    PyObject *m;

    pypmc_Type.tp_new = PyType_GenericNew; 
    if (PyType_Ready(&pypmc_Type) < 0) return NULL;

    m = PyModule_Create(&pypmc_module);

    // Ensure that the module was correctly initialized.
    if (m == NULL) return NULL;

    // Protect the python type from being prematurely garbage collected.
    Py_INCREF(&pypmc_Type);

    // Load module into the interpreter.
    PyModule_AddObject(m, "PyPMC", (PyObject *) &pypmc_Type);

    // Initialize numpy support.
    import_array();

    return m;
}
#elif PYTHON == 2
initpypmc(void)
{
    PyObject *m;

    // Verify that the python type is well-formed.
    if (PyType_Ready(&pypmc_Type) < 0) return;

    m = Py_InitModule3("pypmc", module_methods,
                       "PMC bindings for Python.");

    // Ensure that the module was correctly initialized.
    if (m == NULL) return;

    // Protect the python type from being prematurely garbage collected.
    Py_INCREF(&pypmc_Type);

    // Load module into the interpreter.
    PyModule_AddObject(m, "PyPMC", (PyObject *) &pypmc_Type);

    // Initialize numpy support.
    import_array();
}
#endif
