#pragma once
#include "Detector.hpp"
#include "VideoInpainter.hpp"
#include "Streamer.hpp"
#include "Utilities.hpp"

namespace vm {

	enum class SourceType
	{
		File,
		Camera,
		Gstreamer
	};

	enum class MaskSourceType
	{
		File,
		ObjectDetection
	};

	enum class ManipulationMethod
	{
		Template,
		Inpainting
	};

	enum class TemplateShape
	{
		FromFile,
		Circular,
		Rectangular
	};

	struct ManipulationParams
	{
		cv::Size dimensions = cv::Size(1280, 720);
		int framerate = 30;
		std::string srcPath;
		std::string targetIp;
		int port = 5004;
		std::string maskPath;
		std::string templateSrcPath;
		std::string templateMaskSrcPath;
	};

	class VideoManipulator
	{
	public:
		VideoManipulator();
		~VideoManipulator();

		bool Init(ManipulationParams& parameters);
		bool Init(int argc, char * argv[]);

		void Run();

	private:
		bool first = false;
		bool inpainted = false;
		//Params
		util::MediaType mediaType;
		SourceType srcType;
		MaskSourceType maskSrcType;
		ManipulationMethod manipulationMethod;
		TemplateShape templateShape;

		std::string pathToSrc;
		std::string pathToMask;
		std::string pathToTemplateSrc;
		std::string pathToTemplateMask;
		std::string pathToStoreSrc;
		std::string pathToStoreDetections;
		std::string pathToStoreMask;
		std::string pathToStoreResult;
		std::string targetIp;
		int port;
		cv::Size dimensions;

		bool initialized;
		
		cv::VideoCapture cap;
		cv::VideoWriter writerResult;
		cv::VideoWriter writerSrc;
		cv::VideoWriter writerDetections;
		cv::VideoWriter writerMask;
		cv::VideoWriter WriterGstreamer;

		stream::GStreamer streamer;
		object_detection::Detector detector;
		inpainting::VideoInpainter inpainter;

		bool ValidateParams(ManipulationParams& parameters);
		void InitGStreamer(ManipulationParams& parameters);
		void ProcessImage();
		cv::Mat1b GetMask(cv::Mat3b img);
		cv::Mat3b ManipulateImage(cv::Mat3b img, cv::Mat1b mask);

		int brightenDetectionBy = 100;
		int darkenTemplateBy = 50;

		std::vector<int> detectionTimes;
		std::vector<int> manipulationTimes;
		std::vector<int> totalTimes;
	};
}