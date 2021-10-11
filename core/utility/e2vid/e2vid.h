
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

    void ae_vector_to_numpy(vector<AE> &events, PyArrayObject *&py_mat);

  public:
    // TODO: default values
    bool init(int sensor_height, int sensor_width, int window_size, float events_per_pixel);
    bool predict_grayscale_frame(vector<AE> &input, cv::Mat &output);
};

}