#include <Python.h>
#include "structmember.h"
#include "main.h"

typedef struct {
    PyObject_HEAD

    PyObject *py_parameters;
    PyObject *py_results;

    ExecConfig conf;
    Simulation sim;
    GPUMemory gmem;
} PyPMC;

////////////////////////////////////////////////////////////////////
//// Fundamental methods
static void
pypmc_dealloc( PyPMC *self )
{
    // Call tMCimg's own cleaning up procedure.
    free_mem(self->sim, self->gmem);

    // Free memory used by the python arrays.
    Py_XDECREF(self->py_parameters);
    Py_XDECREF(self->py_results);

    // Finally, delete the pypmc object itself.
    self->ob_type->tp_free((PyObject *) self);
}

static PyObject *
pypmc_new( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
    PyPMC *self = (PyPMC *) type->tp_alloc(type, 0);

    if (self != NULL)
    {
        self->py_parameters = Py_BuildValue("[]");
        if (self->py_parameters == NULL)
        {
            Py_DECREF(self);
            return NULL;
        }
        self->py_results = Py_BuildValue("[]");
        if (self->py_results == NULL)
        {
            Py_DECREF(self);
            return NULL;
        }
    }

    return (PyObject *) self;
}

static int
pypmc_init( PyPMC *self, PyObject *args )
{
    const char *input_filepath;
    int n_threads_per_block, n_threads, n_iterations;

    // The user may optionally pass the .inp filepath to automatically load
    // the simulation parameters from there.
    if(! PyArg_ParseTuple(args, "siii", &input_filepath, &n_threads_per_block, &n_threads, &n_iterations))
        return -1;

    if(input_filepath == NULL)
        return 0;

    // Parse .inp file into the simulation structure.
    read_input(&self->conf, &self->sim, input_filepath);

    parse_conf(&self->conf, n_threads_per_block, n_threads, n_iterations);

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

    if (! PyArg_ParseTuple(args, "|s", &output_filepath))
        return NULL;

    if (output_filepath == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "missing output filepath");
        return NULL;
    }

    write_results(self->sim, output_filepath);

    Py_RETURN_NONE;
}

static PyObject *
pypmc_run( PyPMC *self, PyObject *args )
{
    // Run simulations on the GPU.
    simulate(self->conf, self->sim, self->gmem);

    // TODO: this should be optional
    // Retrieve results to host.
    retrieve(&self->sim, &self->gmem);

    Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////
//// Miscellaneous/Extension-related
static PyMemberDef pypmc_members[] = {
    {"parameters", T_OBJECT_EX, offsetof(PyPMC, py_parameters), 0,
    "list of simulation parameters"},
    {"results", T_OBJECT_EX, offsetof(PyPMC, py_results), 0,
    "list of results"},

    {NULL, NULL, 0, NULL} /* Sentinel */
};

static PyMethodDef pypmc_methods[] = {
    {"run_simulation", (PyCFunction) pypmc_run, METH_NOARGS,
     "Does what it says on the tin."},
    {"write_to_disk", (PyCFunction) pypmc_write_to_disk, METH_VARARGS,
     "Save simulation results to disk, the old-fashioned way."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

static PyTypeObject pypmc_Type = {
    PyObject_HEAD_INIT(NULL)
    0,                                          /* ob_size */
    "pypmc.pypmc",                       /* tp_name */
    sizeof(PyPMC),                       /* tp_basicsize */
    0,                                          /* tp_itemsize */
    (destructor) pypmc_dealloc,            /* tp_dealloc */
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
    "pypmc objects",                       /* tp_doc */
    0,                                          /* tp_traverse */
    0,                                          /* tp_clear */
    0,                                          /* tp_richcompare */
    0,                                          /* tp_weaklistoffset */
    0,                                          /* tp_iter */
    0,                                          /* tp_iternext */
    pypmc_methods,                         /* tp_methods */
    pypmc_members,                         /* tp_members */
    0,                                          /* tp_getset */
    0,                                          /* tp_base */
    0,                                          /* tp_dict */
    0,                                          /* tp_descr_get */
    0,                                          /* tp_descr_set */
    0,                                          /* tp_dictoffset */
    (initproc) pypmc_init,                 /* tp_init */
    0,                                          /* tp_alloc */
    pypmc_new,                             /* tp_new */
};

PyMODINIT_FUNC
initpypmc(void)
{
    // Verify that the python type is well-formed.
    if (PyType_Ready(&pypmc_Type) < 0) return;

    PyObject *m = Py_InitModule3("pypmc", module_methods,
                                 "PMC bindings for Python.");

    // Ensure that the module was correctly initialized.
    if (m == NULL) return;

    // Protect the python type from being prematurely garbage collected.
    Py_INCREF(&pypmc_Type);

    // Load module into the interpreter.
    PyModule_AddObject(m, "pypmc", (PyObject *) &pypmc_Type);
}
