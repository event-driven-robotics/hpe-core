
#include <Python.h>


int main(int argc, char *argv[]) {

    Py_Initialize();

    printf("python home: %s\n", Py_GetPythonHome());
    printf("program name: %s\n", Py_GetProgramName());
    printf("get path: %s\n", Py_GetPath());
    printf("get prefix: %s\n", Py_GetPrefix());
    printf("get exec prefix: %s\n", Py_GetExecPrefix());
    printf("get prog full path: %s\n", Py_GetProgramFullPath());

    PyRun_SimpleString("import sys");
    printf("path: ");
    PyRun_SimpleString("print(sys.path)");
    printf("version: ");
    PyRun_SimpleString("print(sys.version)");

    return 0;
}
