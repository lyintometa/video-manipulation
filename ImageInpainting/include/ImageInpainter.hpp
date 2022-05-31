#pragma once

#include "..\src\InpaintingLevel.hpp"
#include <vector>

namespace inpainting 
{
	class ImageInpainter
	{
	public:
		ImageInpainter();
		~ImageInpainter();

		virtual void Init(InpaintingParams& parameters);
		virtual void Inpaint(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted);

	protected:
		InpaintingParams params;
		std::vector<InpaintingLevel> levels;
		cv::Mat3b mColor;
		cv::Mat1b mAlpha;
		virtual void Initialize(cv::InputArray color, cv::InputArray mask);
		int CalcNumberOfLevels(cv::InputArray color);
		void FillInLowerLv(InpaintingLevel& pmUpper, InpaintingLevel& pmLower);
		void BlendBorder(cv::OutputArray dst);
	};
}