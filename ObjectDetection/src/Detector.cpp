#include "../include/Detector.hpp"
#include "Directory.hpp"
#include <fstream>

const std::string projectDirectory = "\\ObjectDetection\\";

const std::string pathToYolov4Weights("Weights\\yolov4-obj_50000.weights");
const std::string pathToYolov4Config("Models\\yolov4-obj.cfg");
const std::string pathToSignNames("Names\\signs.names");

namespace object_detection {

	Detector::Detector() { }
	Detector::~Detector() { }

	void Detector::Init(DetectionParams& parameters) {
		params = parameters;

		std::string directory = util::GetExeDirectory() + projectDirectory;
		std::string fullPathToSignNames = directory + pathToSignNames;
		std::string fullPathToYolov4Config = directory + pathToYolov4Config;
		std::string fullPathToYolov4Weights = directory + pathToYolov4Weights;

		//Load class names
		std::ifstream ifs(fullPathToSignNames.c_str());
		std::string line;
		while (std::getline(ifs, line)) classes.push_back(line);

		//Load the network
		net = cv::dnn::readNetFromDarknet(fullPathToYolov4Config, fullPathToYolov4Weights);
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}

	void Detector::DetectObjects(cv::InputArray image, int brighterBy) {
		img = image.getMat().clone();
		auto brighter = img + cv::Scalar(brighterBy, brighterBy, brighterBy);
		//Generate 4D blob
		cv::Mat blobFromImg;
		double scaleFactor = 1.0 / 255.0;
		cv::Size size = cv::Size((int)params.resolution, (int)params.resolution);
		cv::Scalar mean = cv::Scalar(0, 0, 0);
		bool swapRB = false;
		bool crop = false;
		cv::dnn::blobFromImage(brighter, blobFromImg, scaleFactor, size, mean, swapRB, crop);

		//Get names of output layers
		std::vector<std::string> outputNames;
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
		std::vector<std::string> layerNames = net.getLayerNames();
		outputNames.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i) {
			outputNames[i] = layerNames[outLayers[i] - 1];
		}

		//Forward blob and names to the network
		std::vector<cv::Mat> netOutput;
		net.setInput(blobFromImg);
		net.forward(netOutput, outputNames);

		//Filter bounding boxes by confidence
		boxes.clear();
		classIds.clear();
		confidences.clear();
		for (size_t i = 0; i < netOutput.size(); ++i) {
			for (int j = 0; j < netOutput[i].rows; ++j) {
				cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
				cv::Point classId;
				double confidence;

				cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
				if (confidence <= params.confThreshold) continue;

				cv::Rect box;
				int centerX, centerY;

				centerX = (int)(netOutput[i].at<float>(j, 0) * img.cols);
				centerY = (int)(netOutput[i].at<float>(j, 1) * img.rows);
				box.width = (int)(netOutput[i].at<float>(j, 2) * img.cols);
				box.height = (int)(netOutput[i].at<float>(j, 3) * img.rows);
				box.x = centerX - box.width / 2;
				box.y = centerY - box.height / 2;

				boxes.push_back(box);
				classIds.push_back(classId.x);
				confidences.push_back((float)confidence);
			}
		}

		//Filter overlapping bounding boxes
		indices.clear();
		cv::dnn::NMSBoxes(boxes, confidences, params.confThreshold, params.nmsThreshold, indices);
	}

	std::vector<std::string> Detector::GetDetectableClasses()
	{
		return classes;
	}

	cv::Mat3b Detector::GetDrawnObjects() {
		//Draw boxes and class names
		cv::Mat image = img.clone();
		for (size_t i = 0; i < indices.size(); ++i) {
			int index = indices[i];
			cv::Rect bbox = boxes[index];
			cv::Scalar bboxColor = cv::Scalar(255, 0, 255);
			cv::rectangle(image, bbox, bboxColor, 2, 8, 0);
			std::stringstream text;
			text << classes[classIds[index]] << " " << (int)(confidences[index] * 100);
			cv::putText(image, text.str(), cv::Point(bbox.x + 3, bbox.y - 5), cv::FONT_HERSHEY_PLAIN, 1, bboxColor, 2, false);
		}
		return image;
	}

	cv::Mat1b Detector::GetObjectMask() {
		cv::Mat1b mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		for (size_t i = 0; i < indices.size(); ++i) {
			int index = indices[i];
			cv::Rect bbox = boxes[index];
			cv::rectangle(mask, bbox, 255, -1);
		}
		return mask;
	}

	cv::Mat1b Detector::GetObjectMaskForType(std::string type) {
		cv::Mat1b mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		for (size_t i = 0; i < indices.size(); ++i) {
			int index = indices[i];
			if (classes[classIds[index]] != type) {
				continue;
			}
			cv::Rect bbox = boxes[index];
			cv::rectangle(mask, bbox, 255, -1);
		}
		return mask;
	}

	cv::Mat3b Detector::ReplaceObjects(cv::Mat3b imageWithObject, cv::Mat1b objectMask, int darkerBy) {
		cv::Mat3b image = img.clone();
		cv::Mat3b darker = imageWithObject - cv::Scalar(darkerBy, darkerBy, darkerBy);

		cv::Mat3b resizedImageWithObject;
		cv::Mat1b resizedObjectMask;

		for (size_t i = 0; i < indices.size(); ++i) {
			int index = indices[i];
			cv::Rect bbox = boxes[index];
			cv::Size bboxSize = bbox.size();
			cv::resize(darker, resizedImageWithObject, bboxSize);
			cv::resize(objectMask, resizedObjectMask, bboxSize);

			for (int j = 0; j < resizedImageWithObject.rows; j++) {
				for (int k = 0; k < resizedImageWithObject.cols; k++) {
					if (resizedObjectMask.at<uchar>(j, k) == 0) continue;
					image.at<cv::Vec3b>(bbox.y + j, bbox.x + k) = resizedImageWithObject.at<cv::Vec3b>(j, k);
				}
			}
		}
		return image;
	}
}