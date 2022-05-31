#include "..\include\Streamer.hpp"
#include "Error.hpp"



namespace stream 
{
	const std::string noTargetIpAndPortMsg = "No target IP address specified at GStreamer initialization";
	const std::string noTargetIpMsg = "No target IP address specified at GStreamer initialization";
	const std::string noPortMsg = "No port specified at GStreamer initialization";
	const std::string couldNotOpen = " could not be opened";

	GStreamer::GStreamer()	{ }
	GStreamer::~GStreamer()	{ }

	void GStreamer::Init(StreamingParams &parameters)
	{
		try
		{
			ValidateParams(parameters);
		}
		catch (const err::ParamsException& e)
		{
			std::cout << "[ERROR] " << e.what() << std::endl;
		}

		dimensions = parameters.dimensions;
		framerate = parameters.framerate;
	}

	cv::VideoCapture GStreamer::OpenCamera()
	{
		std::stringstream ss;
		ss << "ksvideosrc ! videoconvert ! ";
		ss << GetCapsFilter();
		ss << " ! appsink";
		std::cout << ss.str() << std::endl;
		cv::VideoCapture cap(ss.str(), cv::CAP_GSTREAMER);
		if (!cap.isOpened()) err::Exit("Camera" + couldNotOpen);
		return cap;
	}

	cv::VideoCapture GStreamer::OpenTestSrc()
	{
		std::stringstream ss;
		ss << "videotestsrc ! videoconvert ! ";
		ss << GetCapsFilter();
		ss << " ! appsink";
		std::cout << ss.str() << std::endl;
		cv::VideoCapture cap(ss.str(), cv::CAP_GSTREAMER);
		if (!cap.isOpened()) err::Exit("TestSource" + couldNotOpen);
		return cap;
	}

	cv::VideoCapture GStreamer::OpenReceiver(int port)
	{
		if (port = 0) err::Exit(noPortMsg);

		std::stringstream ss;
		ss << "udpsrc port=" << port << " ! ";
		ss << "application/x-rtp, clock-rate=90000, media=video, encoding-name=VP8-DRAFT-IETF-01, tune=zerolatency ! ";
		ss << "rtpvp8depay !vp8dec !videoconvert !appsink";
		std::cout << ss.str() << std::endl;
		cv::VideoCapture cap(ss.str(), cv::CAP_GSTREAMER);
		if (!cap.isOpened()) err::Exit("Receiver" + couldNotOpen);
		return cap;
	}

	cv::VideoWriter GStreamer::OpenTestSink()
	{
		cv::VideoWriter out;
		out.open("appsrc ! videoconvert ! autovideosink ", cv::CAP_GSTREAMER, 0, framerate, dimensions, true);
		if (!out.isOpened()) err::Exit("TestSink" + couldNotOpen);
		return out;
	}

	cv::VideoWriter GStreamer::OpenSender(std::string targetIp, int port)
	{
		if (port = 0 && targetIp.empty()) err::Exit(noTargetIpAndPortMsg);
		if (port = 0) err::Exit(noPortMsg);
		if (targetIp.empty()) err::Exit(noTargetIpMsg);

		std::stringstream ss;
		ss << "appsrc ! videoconvert ! vp8enc ! rtpvp8pay ! queue ! ";
		ss << "udpsink host=" << targetIp << "port=" << port;
		std::cout << ss.str() << std::endl;
		cv::VideoWriter out;
		out.open(ss.str(), 0, framerate, dimensions, true);
		if (!out.isOpened()) err::Exit("Sender" + couldNotOpen);
		return out;
	}

	void GStreamer::ValidateParams(StreamingParams &parameters)
	{
		const std::string moduleName = typeid(GStreamer).name();
		if (parameters.dimensions.height == 0 || parameters.dimensions.width == 0)
			throw err::ParamsException("dimensions", moduleName);
	}

	std::string GStreamer::GetCapsFilter()
	{
		std::stringstream ss;
		ss << "video/x-raw,width=" << dimensions.width << ",height=" << dimensions.height << ",framerate=" << framerate << "/1";
		return ss.str();
	}
}