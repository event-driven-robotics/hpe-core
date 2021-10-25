
#pragma once

#include "utility.h"
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <Python.h>


namespace hpecore {

class E2Vid {

    // python binding
    PyObject *m_py_module;
    PyObject *m_py_fun_init_model;
    PyObject *m_py_fun_predict_grayscale_frame;
    bool m_py_functions_loaded;

    int m_argc;
    char* m_argv[1];

    // functions for converting from c++ to python and viceversa
    template <typename T>
    void ae_vector_to_numpy(std::vector<T> &events, PyArrayObject *&py_mat)
    {
        // create an array of appropriate datatype
        double *data = new double[events.size()];

        // copy the data from the cv::Mat object into the array
        std::memcpy(data, events.data(), events.size() * sizeof(double));

        // the dimensions of the matrix
        npy_intp mdim[] = { events.size(), 4 };

        // convert the vector<AE> to numpy.array
        py_mat = (PyArrayObject *) PyArray_SimpleNewFromData(2, mdim, NPY_FLOAT64, (void*) data);
        PyArray_ENABLEFLAGS(py_mat, NPY_ARRAY_OWNDATA);
    }

    void numpy_to_1Ch_cvMat(PyObject *py_mat, cv::Mat &cv_mat);

    bool init_model(int sensor_height, int sensor_width, int window_size, float events_per_pixel);

    bool python_predict_grayscale_frame(PyArrayObject &events, PyObject &grayscale_frame)
    {
        PyObject* args = Py_BuildValue("(O)", events);
        if(args == NULL)
        {
            std::cout << "args are null" << std::endl;
            return false;
        }

        // execute the function
        PyObject* py_res = PyEval_CallObject(m_py_fun_predict_grayscale_frame, args);

        std::cout << "PyEval_CallObject" << std::endl;

        int ok;
        int res_code = 0;
        ok = PyArg_ParseTuple(py_res, "iO", &res_code, &grayscale_frame);

        Py_XDECREF(args);

        return true;  // TODO: check res_code value!!!
    }

  public:
    // TODO: default values?
    bool init(int sensor_height, int sensor_width, int window_size, float events_per_pixel);
    void close();

    template <typename T>
    bool predict_grayscale_frame(std::vector<T> &input, cv::Mat &output)
    {
        // convert events to numpy array
        PyArrayObject *py_events;
        ae_vector_to_numpy(input, py_events);
        if(py_events == NULL)
        {
            std::cout << "numpy array of events is null" << std::endl;
            return false;
        }

        PyObject* py_grayscale_frame;
        python_predict_grayscale_frame(*py_events, *py_grayscale_frame);

        // convert result
        numpy_to_1Ch_cvMat(py_grayscale_frame, output);

        // decrement object references
        Py_XDECREF(py_grayscale_frame);
        Py_XDECREF(py_events);

        return true;  // TODO: check res_code value!!!
    }

};

}