
#pragma once

#include <Python.h>


class E2Vid {

    // python binding
    PyObject *m_py_module;
    PyObject *m_py_fun_init_model;
    bool m_py_functions_loaded;

    int m_argc;
    char* m_argv[1];

  public:
    bool init_model(int sensor_height, int sensor_width, int window_size, double events_per_pixel);
    bool init(int sensor_height, int sensor_width, int window_size, double events_per_pixel);
};
