#include "../include/VideoInpainter.hpp"


namespace inpainting {

	VideoInpainter::VideoInpainter() : ImageInpainter() { }
	VideoInpainter::~VideoInpainter() { }

	void VideoInpainter::Init(InpaintingParams& parameters) {
		ImageInpainter::Init(parameters);
	}

	void VideoInpainter::Inpaint(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		if (initializedSize != color.size()) {
			Initialize(color, mask);
		}

		LoadFrame(color, mask);

		for (int i = int(levels.size()) - 1; i >= 0; --i)
		{
			levels[i].Run();
			if (i > 0) FillInLowerLv(levels[i], levels[i - 1]);
		}

		BlendBorder(inpainted);
	}

	void VideoInpainter::Initialize(cv::InputArray color, cv::InputArray mask)
	{
		// build pyramid
		initializedSize = color.size();
		levels.resize(CalcNumberOfLevels(color));
		auto size = color.size();

		for (size_t i = 0; i < levels.size(); ++i)
		{
			levels[i].Init(size, params);
			size /= 2;
		}
	}

	void VideoInpainter::LoadFrame(cv::InputArray color, cv::InputArray mask)
	{
		levels[0].LoadFrame(color.getMat(), mask.getMat());

		for (size_t i = 1; i < levels.size(); ++i)
		{
			auto size = levels[i - 1].GetColorPtr()->size() / 2;

			// color
			cv::Mat3b tmpColor;
			cv::resize(*(levels[i - 1].GetColorPtr()), tmpColor, size, 0.0, 0.0, cv::INTER_LINEAR);
			// mask
			cv::Mat1b tmpMask;
			cv::resize(*(levels[i - 1].GetMaskPtr()), tmpMask, size, 0.0, 0.0, cv::INTER_LINEAR);

			levels[i].LoadFrame(tmpColor, tmpMask);
		}

		mColor = color.getMat().clone();
		cv::blur(mask, mAlpha, cv::Size(params.blurSize, params.blurSize));
	}
}