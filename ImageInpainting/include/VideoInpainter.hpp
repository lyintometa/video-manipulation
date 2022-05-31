#pragma once

#include "..\src\InpaintingLevel.hpp"
#include "ImageInpainter.hpp"
#include <vector>

namespace inpainting
{
	class VideoInpainter : ImageInpainter
	{
	public:
		VideoInpainter();
		~VideoInpainter();

		virtual void Init(InpaintingParams& parameters) override;
		virtual void Inpaint(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted) override;

	protected:
		cv::Size initializedSize;
		cv::Mat3b mColorLast;
		cv::Mat1b mAlphaLast;
		virtual void Initialize(cv::InputArray color, cv::InputArray mask) override;
		void LoadFrame(cv::InputArray color, cv::InputArray mask);
	};
}