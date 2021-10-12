
#pragma once

#include "utility.h"
#include <event-driven/all.h>
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
#include <Python.h>
#include <vector>

using namespace ev;


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
    void ae_vector_to_numpy(deque<AE> &events, PyArrayObject *&py_mat);
    void numpy_to_1Ch_cvMat(PyObject *py_mat, cv::Mat &cv_mat);

    // TODO: make the path of the trained model a parameter
    bool init_model(int sensor_height, int sensor_width, int window_size, float events_per_pixel);

  public:
    // TODO: default values?
    bool init(int sensor_height, int sensor_width, int window_size, float events_per_pixel);
    void close();
    bool predict_grayscale_frame(deque<AE> &input, cv::Mat &output);
};

}