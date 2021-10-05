
#include "e2vid.h"
#include <iostream>


using namespace hpecore;

bool E2Vid::init(int sensor_height, int sensor_width, int window_size, float events_per_pixel, bool use_gpu)
{
    m_argc = 1;
    m_argv[0] = strdup("crack_detector");

    // init python interpreter
    Py_Initialize();

    // // print python info
    // printf("python home: %s\n", Py_GetPythonHome());
    // printf("program name: %s\n", Py_GetProgramName());
    // printf("get path: %s\n", Py_GetPath());
    // printf("get prefix: %s\n", Py_GetPrefix());
    // printf("get exec prefix: %s\n", Py_GetExecPrefix());
    // printf("get prog full path: %s\n", Py_GetProgramFullPath());

    // PyRun_SimpleString("import sys");
    // printf("path: ");
    // PyRun_SimpleString("print(sys.path)");
    // printf("version: ");
    // PyRun_SimpleString("print(sys.version)");

    // set the command line arguments (can be crucial for some python-packages, like tensorflow)
    // PySys_SetArgv(m_argc, (wchar_t**)m_argv);
    // PySys_SetArgv(m_argc, m_argv);

#ifdef PYTHON2
    PySys_SetArgv(m_argc, m_argv);
#else
    wchar_t ** w_argv = new wchar_t*;
    *w_argv = Py_DecodeLocale(m_argv[0], NULL);
    PySys_SetArgv(m_argc, w_argv);
#endif

    // add scripts folder to the python's PATH
    PyObject *sys_path = PySys_GetObject(strdup("path"));
    PyList_Append(sys_path, PyUnicode_FromString("hpe-core/core/utility/e2vid/python"));  // relative to the build directory

    // important, initializes several of static data structures
    import_array1(-1);

    m_py_functions_loaded = false;

    // load python module
    std::string py_module_name = "run_e2vid";
    m_py_module = PyImport_ImportModule(py_module_name.c_str());
    if(m_py_module == NULL)
    {
        std::cout << "Could not load Python module \"" << py_module_name << "\"!" << std::endl;
        return false;
    }

    // get dictionary of available items in the module
    PyObject *m_PyDict = PyModule_GetDict(m_py_module);

    // bind python functions
    std::string py_fun_name = "predict_grayscale_frame";
    m_py_fun_predict_grayscale_frame = PyDict_GetItemString(m_PyDict, py_fun_name.c_str());
    if(m_py_fun_predict_grayscale_frame == NULL)
    {
        std::cout << "Could not load function \"" << py_fun_name << "\" from Python module \"" << py_module_name << "\"!" << std::endl;
        return false;
    }

    m_py_functions_loaded = true;

    // TODO: call a script/function that initializes e2vid model with function parameters

    return true;
}


/*******************/
/* PRIVATE METHODS */
/*******************/

void E2Vid::ae_vector_to_numpy(vector<AE> &events, PyArrayObject *&py_mat)
{

//    // total number of elements (here it's a grayscale 640x480)
//    int nElem = cv_mat.rows * cv_mat.cols;
//
//    // create an array of apropriate datatype
//    uchar *data = new uchar[nElem];
//
//    // copy the data from the cv::Mat object into the array
//    std::memcpy(data, cv_mat.data, nElem * sizeof(uchar));
//
//    // the dimensions of the matrix
//    npy_intp mdim[] = { cv_mat.rows, cv_mat.cols };
//
//    // convert the cv::Mat to numpy.array
//    py_mat = (PyArrayObject *) PyArray_SimpleNewFromData(2, mdim, NPY_UINT8, (void*) data);
//    PyArray_ENABLEFLAGS(py_mat, NPY_ARRAY_OWNDATA);
}


/******************/
/* PUBLIC METHODS */
/******************/

bool E2Vid::predict_grayscale_frame(vector<AE> &input, cv::Mat &output)
{
//    // - convert events to numpy arr
//    // - initialize output
//
//    if(!m_py_functions_loaded)
//    {
//        std::cout << "python functions not loaded" << std::endl;
//        return false;
//    }
//
//    // - send events to python
//
//    // create a Python-tuple of arguments for the function call
//    PyObject* args = Py_BuildValue("(OOsiiOOdd)",
//                                   py_images,
//                                   py_tile_mask,
//                                   model_root.c_str(),
//                                   m_patch_dimension,
//                                   m_stride,
//                                   m_infer_in_batch ? Py_True : Py_False,
//                                   m_probability_based_inference ? Py_True : Py_False,
//                                   m_prob_cutoff,
//                                   m_tile_th
//                                   );
//
//    if(args == NULL)
//    {
//        Py_XDECREF(py_images);
//        Py_XDECREF(py_tile_mask);
//        std::cout << "args are null" << std::endl;
//        return false;
//    }
//
//    std::cout << "args" << std::endl;
//
//    // execute the function
//    PyObject* py_res = PyEval_CallObject(m_py_fun_detect_cracks_locally_multi_images, args);
//
//    std::cout << "PyEval_CallObject" << std::endl;
//
//    int ok;
//    int res_code = 0;
//    PyObject* py_crack_mask;
//    ok = PyArg_ParseTuple(py_res, "iO", &res_code, &py_crack_mask);
//
//    // convert result
//    Numpy_to_1Ch_cvMat(py_crack_mask, crack_mask);
//
//    // decrement the object references
//    Py_XDECREF(py_crack_mask);
//    Py_XDECREF(py_images);
//    Py_XDECREF(py_tile_mask);
//    Py_XDECREF(args);

    return true;  // TODO: check res_code value!!!
}