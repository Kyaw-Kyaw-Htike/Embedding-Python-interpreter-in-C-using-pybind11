#include <vector>

#undef slots // only needed if using Qt
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#define slots Q_SLOTS  // only needed if using Qt

namespace py = pybind11;
using namespace py::literals;

namespace {
	
class PyGuard {
public:
    PyGuard(std::string dir_pythonHome) {
        Py_SetPythonHome(std::wstring(dir_pythonHome.begin(), dir_pythonHome.end()).c_str());
        pybind11::initialize_interpreter();
    }
    ~PyGuard() {
        pybind11::finalize_interpreter();
    }
};

void add_localsPy_to_globalsPy(const py::dict& locals_py) {
    py::dict globals_py = py::globals();
    for (const auto item : locals_py) {
        globals_py[item.first] = item.second;
    }
}

}



int main(int argc, char *argv[])
{
	std::string dir_pythonHome = "C:/Users/alikyaw/Anaconda3/envs/pytorch";

	// *********************** begin: python init *********************************** //
	
	Py_SetPythonHome(std::wstring(dir_pythonHome.begin(), dir_pythonHome.end()).c_str());
	pybind11::initialize_interpreter();
	
	py::dict locals_py;
	locals_py["numA"] = 256;
    locals_py["numB"] = 1000;
	
    py::exec(R"(
	
	someVar = numA * 10
	AnotherVar = numB // 10
 
	def adder(num1, num2):
		return num1 + num2 + someVar + AnotherVar
        
    )", py::globals(), locals_py);

    add_localsPy_to_globalsPy(locals_py);
	
	// *********************** end: python init *********************************** //
	
	for (int i=50; i<100; i++) {
		for (int j=74l; j<1999; j++) {
				int result_adder = locals_py["adder"](i, j).cast<int>();
				printf("calling python function adder(%d, %d) gives %d\n", i, j, result_adder);
		}
	}

	
	pybind11::finalize_interpreter();
		
    return 0;
}
