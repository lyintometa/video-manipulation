#include "../include/VideoManipulator.hpp"
#include "Utilities.hpp"
#include "Error.hpp"
#include <iostream>
#include <chrono>

namespace vm
{
	const std::string port = "port=";

	VideoManipulator::VideoManipulator() { }
	VideoManipulator::~VideoManipulator() { }

	bool VideoManipulator::Init(ManipulationParams& parameters)
	{
		if (!ValidateParams(parameters))
		{
			std::cout << "[ERROR] Invalid manipulation parameters, initialization canceled!";
			return false;
		}

		pathToSrc = parameters.srcPath;
		pathToStoreResult = util::GetFullNameResult(pathToSrc);
		if (!pathToSrc.empty())
		{
			srcType = SourceType::File;
			mediaType = util::GetMediaType(pathToSrc);
			//pathToStoreSrc = "";
			pathToStoreSrc = util::GetFullNameCopy(pathToSrc);
			cap = cv::VideoCapture(pathToSrc);
			if (!cap.isOpened()) err::Exit("Video file could not be opened");
		}
		else
		{
			InitGStreamer(parameters);
			srcType = SourceType::Gstreamer;
			mediaType = util::MediaType::Video;
			pathToStoreSrc = util::GetFullNameCopy();
			cap = streamer.OpenReceiver(parameters.port);
		}

		pathToMask = parameters.maskPath;
		if (!pathToMask.empty())
		{
			maskSrcType = MaskSourceType::File;
			pathToStoreMask = "";
			pathToStoreDetections = "";
		}
		else
		{
			maskSrcType = MaskSourceType::ObjectDetection;
			pathToStoreMask = util::GetFullNameGeneratedMask(pathToSrc);
			pathToStoreDetections = util::GetFullNameDetections(pathToSrc);

			object_detection::DetectionParams detectParams;
			detectParams.resolution = object_detection::RESOLUTION::HIGH;
			detectParams.confThreshold = 0.3f;
			detectParams.nmsThreshold = 0.2f;
			detector.Init(detectParams);
		}

		pathToTemplateSrc = parameters.templateSrcPath;
		if (!pathToTemplateSrc.empty())
		{
			manipulationMethod = ManipulationMethod::Template;
		}
		else
		{
			manipulationMethod = ManipulationMethod::Inpainting;
			inpainting::InpaintingParams inpaintParams;
			inpaintParams.alpha = 0.15f;			// 0.0f means no spatial cost considered
			inpaintParams.beta = 0.99f;			    // 0.0f means no coherence cost considered
			inpaintParams.maxItr = 1;				// set to 1 to crank up the speed
			inpaintParams.maxRandSearchItr = 1;	// set to 1 to crank up the speed
			inpainter.Init(inpaintParams);
		}

		pathToTemplateMask = parameters.templateMaskSrcPath;
		if (!pathToTemplateMask.empty())
		{
			templateShape = TemplateShape::FromFile;
		}
		else
		{
			templateShape = TemplateShape::Circular;
		}

		targetIp = parameters.targetIp;
		if (!targetIp.empty()) {
			InitGStreamer(parameters);
			WriterGstreamer = streamer.OpenSender(parameters.targetIp, parameters.port);
		}

		dimensions = parameters.dimensions;

		initialized = true;
		return true;
	}

	bool VideoManipulator::Init(int argc, char * argv[])
	{
		ManipulationParams parameters;

		const std::string src = "src=";
		const std::string mask = "mask=";
		const std::string port = "port=";
		const std::string targetIp = "targetip=";
		const std::string width = "width=";
		const std::string height = "height=";
		const std::string templateSrc = "template=";
		const std::string templateMaskSrc = "templateMask=";

		for (int i = 0; i < argc; ++i)
		{
			std::string arg = argv[i];

			if (arg.rfind(src, 0)) parameters.srcPath = arg.substr(src.length());
			if (arg.rfind(mask, 0)) parameters.maskPath = arg.substr(src.length());
			if (arg.rfind(port, 0)) parameters.port = std::stoi(arg.substr(port.length()));
			if (arg.rfind(targetIp, 0)) parameters.targetIp = arg.substr(targetIp.length());
			if (arg.rfind(width, 0)) parameters.dimensions.width = std::stoi(arg.substr(width.length()));
			if (arg.rfind(height, 0)) parameters.dimensions.height = std::stoi(arg.substr(height.length()));
			if (arg.rfind(templateSrc, 0)) parameters.templateSrcPath = arg.substr(templateSrc.length());
			if (arg.rfind(templateMaskSrc, 0)) parameters.templateMaskSrcPath = arg.substr(templateMaskSrc.length());
		}

		return Init(parameters);
	}

	void VideoManipulator::Run()
	{
		if (!initialized) return;

		if (mediaType == util::MediaType::Image && targetIp.empty()) {
			ProcessImage();
			return;
		}

		cv::Mat3b frame;
		if (mediaType == util::MediaType::Image) frame = cv::imread(pathToSrc);

		int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
		if (mediaType == util::MediaType::Video)
			writerResult.open(pathToStoreResult, codec, 30, dimensions, frame.type());
		if (!pathToStoreSrc.empty()) writerSrc.open(pathToStoreSrc, codec, 30, dimensions, frame.type());
		if (!pathToStoreDetections.empty()) writerDetections.open(pathToStoreDetections, codec, 30, dimensions, frame.type());
		if (!pathToStoreMask.empty()) writerMask.open(pathToStoreMask, codec, 30, dimensions, frame.type());

		int i = 0;
		while (true) {
			auto startTotal = std::chrono::steady_clock::now();

			if (mediaType == util::MediaType::Video) {
				cap >> frame;
				if (frame.empty())
				{
					break;
				}
			}

			cv::resize(frame, frame, dimensions);
			if(!pathToStoreSrc.empty()) writerSrc.write(frame);
			cv::imshow("Image", frame);

			auto mask = GetMask(frame);
			auto manipulated = ManipulateImage(frame, mask);

			auto endTotal = std::chrono::steady_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(endTotal - startTotal).count();
			std::cout << "Total frame time: " << time << "ms" << std::endl;
			if(i != 0 && inpainted) totalTimes.push_back(time); //exclude first frame

			if (cv::waitKey(10) >= 0) {
				break;
			}
			i++;
			inpainted = false;
			first = true;
		}

		std::cout << "Average detection time: " << util::VectorAverage(detectionTimes) << "ms" << std::endl;
		std::cout << "Average manipulation time: " << util::VectorAverage(manipulationTimes) << "ms" << std::endl;
		std::cout << "Average total time: " << util::VectorAverage(totalTimes) << "ms" << std::endl;
	}

	bool VideoManipulator::ValidateParams(ManipulationParams & parameters)
	{
		bool valid = true;

		if ((parameters.srcPath.empty() && parameters.port == 5004)
			|| (!parameters.targetIp.empty() && parameters.port == 5004)) {
			std::cout << "[WARNING]: Using default port '5004' for GStreamer. Change it by using ''"
				<< port << "' command line argument or change the manipulation parameters!" << std::endl;
		}

		return valid;
	}

	void VideoManipulator::InitGStreamer(ManipulationParams& parameters)
	{
		if (!parameters.srcPath.empty() && targetIp.empty()) return;
		stream::StreamingParams streamParams;
		streamParams.dimensions = parameters.dimensions;
		streamParams.framerate = parameters.framerate;
		streamer.Init(streamParams);
	}

	void VideoManipulator::ProcessImage()
	{
		auto start = std::chrono::steady_clock::now();
		cv::Mat3b img = cv::imread(pathToSrc);
		cv::resize(img, img, dimensions);
		if (!pathToStoreSrc.empty()) cv::imwrite(pathToStoreSrc, img);
		cv::imshow("Image", img);
		auto mask = GetMask(img);
		auto manipulated = ManipulateImage(img, mask);
		auto end = std::chrono::steady_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "Total time: " << time << "ms" << std::endl;
	}

	cv::Mat1b VideoManipulator::GetMask(cv::Mat3b img)
	{
		cv::Mat1b mask;
		cv::Mat3b detected;
		if (maskSrcType == MaskSourceType::File)
		{
			mask = cv::imread(pathToMask, cv::IMREAD_GRAYSCALE) >= 255;
			cv::resize(mask, mask, dimensions);
		}
		else
		{
			auto start = std::chrono::steady_clock::now();
			detector.DetectObjects(img, brightenDetectionBy);
			auto end = std::chrono::steady_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "Detection time: " << time << "ms" << std::endl;
			if(first) detectionTimes.push_back(time);
			detected = detector.GetDrawnObjects();

			mask = detector.GetObjectMask();
			cv::bitwise_not(mask, mask);
		}

		if (!pathToStoreDetections.empty())
		{
			if (mediaType == util::MediaType::Image && targetIp.empty())
				cv::imwrite(pathToStoreDetections, detected);
			else writerDetections.write(detected);
		}

		cv::imshow("Detections", detected);

		if (!pathToStoreMask.empty())
		{
			if (mediaType == util::MediaType::Image && targetIp.empty())
				cv::imwrite(pathToStoreMask, mask);
			else writerMask.write(mask);
		}

		cv::imshow("Mask", mask);
		return mask;
	}

	cv::Mat3b VideoManipulator::ManipulateImage(cv::Mat3b img, cv::Mat1b mask)
	{
		cv::Mat3b manipulated;
		if (manipulationMethod == ManipulationMethod::Template)
		{
			cv::Mat3b templateObject = cv::imread(pathToTemplateSrc);

			cv::Mat1b templateMask = cv::Mat::zeros(templateObject.rows, templateObject.cols, CV_8U);
			if (templateShape == TemplateShape::Circular)
			{
				cv::circle(templateMask, cv::Point2d(templateMask.rows / 2, templateMask.cols / 2), templateMask.rows / 2, 255, -1);
			}
			else if (templateShape == TemplateShape::Rectangular)
			{
				cv::bitwise_not(templateMask, templateMask);
			}
			else if (templateShape == TemplateShape::FromFile)
			{
				templateMask = cv::imread(pathToTemplateMask, cv::IMREAD_GRAYSCALE) >= 255;
			}
			auto start = std::chrono::steady_clock::now();
			manipulated = detector.ReplaceObjects(templateObject, templateMask, darkenTemplateBy);
			auto end = std::chrono::steady_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			std::cout << "Template time: " << time << "ms" << std::endl;
			manipulationTimes.push_back(time);
		}
		else
		{
			auto start = std::chrono::steady_clock::now();
			inpainter.Inpaint(img, mask, manipulated);
			auto end = std::chrono::steady_clock::now();
			auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

			cv::Mat1b invertedMask;
			cv::bitwise_not(mask, invertedMask);
			if (cv::countNonZero(invertedMask) != 0) //only count time if sth to inpaint
			{
				inpainted = true;
				manipulationTimes.push_back(time); 
				std::cout << "Inpainting time: " << time << "ms" << std::endl;
			}
			else
			{
				std::cout << "Inpainting time: " << time << "ms (empty mask)" << std::endl;
			}
		}

		if (mediaType == util::MediaType::Image && targetIp.empty())
			cv::imwrite(pathToStoreResult, manipulated);
		else writerResult.write(manipulated);

		if (!targetIp.empty()) WriterGstreamer.write(manipulated);

		cv::imshow("Manipulated", manipulated);
		return manipulated;
	}
}