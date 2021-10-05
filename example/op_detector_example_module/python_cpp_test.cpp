
#include <Python.h>
#include <hpe-core/e2vid.h>


int main(int argc, char *argv[]) {

    Py_Initialize();

    printf("python home: %ls\n", Py_GetPythonHome());
    printf("program name: %ls\n", Py_GetProgramName());
    printf("get path: %ls\n", Py_GetPath());
    printf("get prefix: %ls\n", Py_GetPrefix());
    printf("get exec prefix: %ls\n", Py_GetExecPrefix());
    printf("get prog full path: %ls\n", Py_GetProgramFullPath());

    PyRun_SimpleString("import sys");
    printf("path: ");
    PyRun_SimpleString("print(sys.path)");
    printf("version: ");
    PyRun_SimpleString("print(sys.version)");

    printf("E2VID_PYTHON_DIR: %s\n", E2VID_PYTHON_DIR);

    return 0;
}
