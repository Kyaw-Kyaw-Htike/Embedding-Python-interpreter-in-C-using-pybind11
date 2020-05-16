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

void perform_nms(std::vector<cv::Rect>& dr_nms, std::vector<float>& ds_nms, const std::vector<cv::Rect>& dr, const std::vector<float>& ds, double thresh_overlap_ratio=0.5) {
    size_t nrects = dr.size();

    // sort the dr and ds based on ds
    std::vector<size_t> idx(nrects);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&ds](size_t i1, size_t i2) {return ds[i1] > ds[i2]; });
    std::vector<cv::Rect> dr_sorted(nrects);
    std::vector<float> ds_sorted(nrects);
    for (size_t i = 0; i < nrects; i++) {
        dr_sorted[i] = dr[idx[i]];
        ds_sorted[i] = ds[idx[i]];
    }

    std::vector<char> suppressed(nrects, 0);
    for (size_t i = 0; i < nrects; i++)
    {
        if (suppressed[i]) continue;
        for (size_t j = i+1; j < nrects; j++)
        {
            double intersection_area = (dr_sorted[i] & dr_sorted[j]).area();
            double union_area = (dr_sorted[i] | dr_sorted[j]).area();
            double overlap_ratio = intersection_area / union_area;
            if (overlap_ratio >= thresh_overlap_ratio) {
                suppressed[j] = 1;
            }
        }
    }

    dr_nms.clear();
    ds_nms.clear();
    dr_nms.reserve(nrects);
    ds_nms.reserve(nrects);
    for (size_t i = 0; i < nrects; i++)
    {
        if (!suppressed[i]) {
            dr_nms.push_back(dr_sorted[i]);
            ds_nms.push_back(ds_sorted[i]);
        }
    }
}

class HeadDetectorTopDownCNNPytorch
{
public:
    HeadDetectorTopDownCNNPytorch(double conf_thresh=0.5);
    ~HeadDetectorTopDownCNNPytorch() override;
    std::vector<cv::Rect> detect(const cv::Mat& frame);
private:
    struct m_p;
    m_p* m_pp;
};

struct HeadDetectorTopDownCNNPytorch::m_p {
    const std::string dir_proj = "G:/Projects/PyTorch-YOLOv3-master/test_code_1";
    const std::string fpath_vid = "F:/Datasets/Head/topDown/vids_640x360/NDC-101.mp4";
    const int imgSize_netInput = 416;
    double conf_thresh;
    const double nms_thresh = 0.5;
    const bool suppress_overlapping_boxes = true;
    const double factor_width = 1.2;
    const double factor_height = 1.2;
    py::dict locals_py;
};

HeadDetectorTopDownCNNPytorch::HeadDetectorTopDownCNNPytorch(double conf_thresh)
{
    m_pp = new m_p();
    m_pp->conf_thresh = conf_thresh;
    m_pp->locals_py["dir_proj"] = m_pp->dir_proj;
    m_pp->locals_py["imgSize_netInput"] = m_pp->imgSize_netInput;
    m_pp->locals_py["conf_thresh"] = m_pp->conf_thresh;

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

    def detect_heads(img_np_rgb_resized_padded):
        image_tensor = torch.from_numpy(img_np_rgb_resized_padded)
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
    )", py::globals(), m_pp->locals_py);

    add_localsPy_to_globalsPy(m_pp->locals_py);
}

HeadDetectorTopDownCNNPytorch::~HeadDetectorTopDownCNNPytorch()
{
    delete m_pp;
}

std::vector<cv::Rect> HeadDetectorTopDownCNNPytorch::detect(const cv::Mat& img_bgr)
{
    int height_img, width_img, imw, imh, pad_leftOrRight, pad_topOrBottom;
    double ratio, mult_factor;
    cv::Size target_imgSize;
    double pad_x, pad_y, unpad_h, unpad_w, unpad_factor_x, unpad_factor_y, pad_x_half, pad_y_half;
    const cv::Scalar pad_fill_color(128, 128, 128);

    height_img = img_bgr.rows;
    width_img = img_bgr.cols;
    ratio = std::min(divP(m_pp->imgSize_netInput, width_img), divP(m_pp->imgSize_netInput, height_img));
    imw = std::round(width_img * ratio);
    imh = std::round(height_img * ratio);
    target_imgSize.width = imw;
    target_imgSize.height = imh;

    pad_leftOrRight = std::max(divP(imh - imw, 2), 0.0);
    pad_topOrBottom = std::max(divP(imw - imh, 2), 0.0);

    mult_factor = divP(m_pp->imgSize_netInput, std::max(height_img, width_img));
    pad_x = std::max(height_img - width_img, 0) * mult_factor;
    pad_y = std::max(width_img - height_img, 0) * mult_factor;
    unpad_h = m_pp->imgSize_netInput - pad_y;
    unpad_w = m_pp->imgSize_netInput - pad_x;
    unpad_factor_x = divP(width_img, unpad_w);
    unpad_factor_y = divP(height_img, unpad_h);
    pad_x_half = pad_x / 2;
    pad_y_half = pad_y / 2;

    cv::Mat img_bgr_resized;
    cv::resize(img_bgr, img_bgr_resized, target_imgSize);
    cv::Mat img_bgr_resized_padded;
    cv::copyMakeBorder(img_bgr_resized, img_bgr_resized_padded, pad_topOrBottom, pad_topOrBottom, pad_leftOrRight, pad_leftOrRight, cv::BORDER_CONSTANT, pad_fill_color);
    cv::Mat img_rgb_resized_padded;
    cv::cvtColor(img_bgr_resized_padded, img_rgb_resized_padded, cv::COLOR_BGR2RGB);
    img_rgb_resized_padded.convertTo(img_rgb_resized_padded, CV_32F, 1/255.0);
    if (!img_rgb_resized_padded.isContinuous())
        img_rgb_resized_padded = img_rgb_resized_padded.clone();

    typedef py::array_t<float, py::array::c_style | py::array::forcecast> NumpyArrayInput;
    typedef py::array_t<float, py::array::c_style | py::array::forcecast> NumpyArrayOutput;

    NumpyArrayInput img_np_rgb_resized_padded({img_rgb_resized_padded.rows, img_rgb_resized_padded.cols, 3}, img_rgb_resized_padded.ptr<float>());
    NumpyArrayOutput detections = m_pp->locals_py["detect_heads"](img_np_rgb_resized_padded).cast<NumpyArrayOutput>();

    int ndets = detections.shape(0);

    std::vector<cv::Rect> bboxes(ndets);
    std::vector<float> scores(ndets);

    for (int i = 0; i < ndets; ++i) {
        float cx = detections.at(i, 0);
        float cy = detections.at(i, 1);
        float w = detections.at(i, 2);
        float h = detections.at(i, 3);
        float conf = detections.at(i, 4);

        // convert (cx, cy, w, h) to (x1, y1, x2, y2)
        float x1 = cx - divP(w, 2) + 0.5;
        float y1 = cy - divP(h, 2) + 0.5;
        float x2 = x1 + w - 1;
        float y2 = y1 + h - 1;

        // convert bbox (x1,y1,x2,y2) to original image space
        float x1_new = (x1 - pad_x_half) * unpad_factor_x;
        float y1_new = (y1 - pad_y_half) * unpad_factor_y;
        float x2_new = (x2 - pad_x_half) * unpad_factor_x;
        float y2_new = (y2 - pad_y_half) * unpad_factor_y;
        float w_new = x2_new - x1_new + 1;
        float h_new = y2_new - y1_new + 1;

        // scale bounding box (due to context added during training)
        float xc_new2 = x1_new + divP(w_new, 2) - 0.5;
        float yc_new2 = y1_new + divP(h_new, 2) - 0.5 ;
        float w_new2 = w_new * divP(1, m_pp->factor_width);
        float h_new2 = h_new * divP(1, m_pp->factor_height);
        float x1_new2 = xc_new2 - divP(w_new2, 2) + 0.5;
        float y1_new2 = yc_new2 - divP(h_new2, 2) + 0.5;
        float x2_new2 = x1_new2 + w - 1;
        float y2_new2 = y1_new2 + h - 1;

        bboxes[i] = cv::Rect(std::round(x1_new2), std::round(y1_new2), std::round(w_new2), std::round(h_new2));
        scores[i] = conf;
    } // end: for (int i = 0; i < ndets; ++i)

    std::vector<cv::Rect> bboxes_nms;
    std::vector<float> scores_nms;

    perform_nms(bboxes_nms, scores_nms, bboxes, scores, m_pp->nms_thresh);

    return bboxes_nms;

}

}



int main(int argc, char *argv[])
{

    PyGuard guard("C:/Users/alikyaw/Anaconda3/envs/pytorch_learn");
	HeadDetectorTopDownCNNPytorch detector;
	cv::Mat img = cv::imread("some.jpg");
	std::vector<cv::Rect> bboxes = detector.detect(img);
		
    return 0;
}