#pragma once

#include <opencv2/opencv.hpp>

namespace object_detection 
{
	enum class RESOLUTION 
	{
		LOW = 320,
		MEDIUM = 416,
		HIGH = 608,
	};

	struct DetectionParams 
	{
		RESOLUTION resolution = RESOLUTION::HIGH;
		float confThreshold = 0.5f;
		float nmsThreshold = 0.5f;
	};

	class Detector
	{
	public:
		Detector();
		~Detector();

		void Init(DetectionParams& parameters);
		void DetectObjects(cv::InputArray image, int brighterBy = 0);
		std::vector<std::string> GetDetectableClasses();
		cv::Mat3b GetDrawnObjects();
		cv::Mat1b GetObjectMask();
		cv::Mat1b GetObjectMaskForType(std::string type);
		cv::Mat3b ReplaceObjects(cv::Mat3b imageWithObject, cv::Mat1b objectMask, int darkerBy = 0);

	private:
		std::vector<std::string> classes;
		cv::dnn::Net net;
		cv::Mat img;
		DetectionParams params;
		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		std::vector<int> indices;
	};
}
