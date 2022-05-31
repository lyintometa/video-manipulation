#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

namespace stream
{
	struct StreamingParams {
		cv::Size dimensions;
		int framerate = 30;
	};

	class GStreamer 
	{
	public:
		GStreamer();
		~GStreamer();

		void Init(StreamingParams &parameters);
		cv::VideoCapture OpenCamera();
		cv::VideoCapture OpenTestSrc();
		cv::VideoCapture OpenReceiver(int port = 5004);
		cv::VideoWriter OpenTestSink();
		cv::VideoWriter OpenSender(std::string targetIp = "localhost", int port = 5004);

	private:
		cv::Size dimensions;
		int framerate;

		void ValidateParams(StreamingParams &parameters);
		std::string GetCapsFilter();
	};
	
}