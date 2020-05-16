#include <vector>
#include "opencv2/opencv.hpp"

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
	std::string dir_pythonHome = "C:/Users/alikyaw/Anaconda3/envs/pytorch_learn";
	std::string fpath_vid = "path_to_video_file.mp4";

	// *********************** begin: python init *********************************** //
	
	Py_SetPythonHome(std::wstring(dir_pythonHome.begin(), dir_pythonHome.end()).c_str());
	pybind11::initialize_interpreter();
	
	py::dict locals_py;
    locals_py["dir_proj"] = "someDir";
	locals_py["imgSize_netInput"] = 256;
    locals_py["conf_thresh"] = 0.5;
	
    py::exec(R"(

    import torch
    import cv2
    import numpy as np
    import sys
    import os

    sys.path.append(dir_proj)
    import models

    config_path=os.path.join(dir_proj, 'config/yolov3-custom.cfg')
    weights_path=os.path.join(dir_proj, 'checkpoints_20000_s7/yolov3_ckpt_3.pth')
    with_gpu = 1

    cuda_is_available = torch.cuda.is_available()
    if not with_gpu:
        cuda_is_available = False

    device = torch.device("cuda" if cuda_is_available else "cpu")

    model = models.Darknet(config_path, img_size=imgSize_netInput).to(device)
    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
    else:
        if with_gpu:
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    if cuda_is_available:
        model.cuda()

    model.eval()
    Tensor = torch.cuda.FloatTensor if cuda_is_available else torch.FloatTensor

    def detect_heads(img_np):
        image_tensor = torch.from_numpy(img_np)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        input_net = torch.autograd.Variable(image_tensor.type(Tensor))
        if not cuda_is_available:
            input_net = input_net.cpu()
        with torch.no_grad():
            detections = model(input_net)
        detections.squeeze_(0)
        detections = detections[detections[:, 4] >= conf_thresh]
        detections = detections.numpy()
        return detections
		
	def adder(num1, num2):
		return num1 + num2
        
    )", py::globals(), locals_py);

    add_localsPy_to_globalsPy(locals_py);
	
	// *********************** end: python init *********************************** //
	
	cv::VideoCapture vid(fpath_vid);
	
	while (true) {
		cv::Mat img;
		vid >> img;
		if (img.empty()) break;
				
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_32F, 1/255.0);
		if (!img.isContinuous())
			img = img.clone();

		typedef py::array_t<float, py::array::c_style | py::array::forcecast> NumpyArrayInput;
		typedef py::array_t<float, py::array::c_style | py::array::forcecast> NumpyArrayOutput;

		NumpyArrayInput img_np({img.rows, img.cols, 3}, img.ptr<float>());
		NumpyArrayOutput detections = locals_py["detect_heads"](img_np).cast<NumpyArrayOutput>();
		
		int result_adder = locals_py["adder"](100, 50).cast<int>();
		printf("result_adder = %d\n", result_adder);
		
		for (int i = 0; i < detections.shape(0); ++i) {
			float x = detections.at(i, 0);
			float y = detections.at(i, 1);
			float w = detections.at(i, 2);
			float h = detections.at(i, 3);
			float conf = detections.at(i, 4);	
			printf("detection %d, x = %f, y = %f, w = %f, h = %f, conf = %f\n", i, x, y, w, h, conf);		
		} 		
	}
	
	pybind11::finalize_interpreter();
		
    return 0;
}