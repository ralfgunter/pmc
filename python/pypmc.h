#ifndef _PYPMC_H_
#define _PYPMC_H_

#include <Python.h>
#include "structmember.h"
#include "main.h"
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <numpy/arrayobject.h>
#include "multithreading.h"

#include "cuda.h"

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

    int gpu_id;
    CUcontext ctx;	// TODO: get rid of this. GET RID OF THIS.

    PyObject *py_path_length, *py_momentum_transfer;
    PyObject *py_medium;
    PyArrayObject *py_fluence;

    ExecConfig conf;
    Simulation sim;
    GPUMemory gmem;
} PyPMC;

static PyObject* pypmc_get_tissueArray( Simulation sim, float *tissueArray );
static PyArrayObject* pypmc_fluence_to_ndarray( Simulation sim, float *fbox );

#endif  // _PYPMC_H_
