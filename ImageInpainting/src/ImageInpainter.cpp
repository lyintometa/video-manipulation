#include "..\include\ImageInpainter.hpp"

namespace inpainting {

	ImageInpainter::ImageInpainter() { }
	ImageInpainter::~ImageInpainter() { }

	void ImageInpainter::Init(InpaintingParams& parameters) {
		params = parameters;
	}

	void ImageInpainter::Inpaint(cv::InputArray color, cv::InputArray mask, cv::OutputArray inpainted)
	{
		assert(color.size() == mask.size());
		assert(color.type() == CV_8UC3);
		assert(mask.type() == CV_8U);

		Initialize(color, mask);

		for (int i = int(levels.size()) - 1; i >= 0; --i)
		{
			levels[i].Run();
			if (i > 0) FillInLowerLv(levels[i], levels[i - 1]);
		}

		BlendBorder(inpainted);
	}

	void ImageInpainter::Initialize(cv::InputArray color, cv::InputArray mask)
	{
		// build pyramid
		levels.resize(CalcNumberOfLevels(color));
		levels[0].Init(color.size(), params);
		levels[0].LoadFrame(color.getMat(), mask.getMat());

		for (int i = 1; i < levels.size(); ++i)
		{
			auto lvSize = levels[i - 1].GetColorPtr()->size() / 2;

			// color
			cv::Mat3b tmpColor;
			cv::resize(*(levels[i - 1].GetColorPtr()), tmpColor, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			// mask
			cv::Mat1b tmpMask;
			cv::resize(*(levels[i - 1].GetMaskPtr()), tmpMask, lvSize, 0.0, 0.0, cv::INTER_LINEAR);
			for (int r = 0; r < tmpMask.rows; ++r)
			{
				auto ptrMask = tmpMask.ptr<uchar>(r);
				for (int c = 0; c < tmpMask.cols; ++c)
				{
					ptrMask[c] = ptrMask[c] < 255 ? 0 : 255;
				}
			}

			levels[i].Init(color.size(), params);
			levels[i].LoadFrame(tmpColor, tmpMask);
		}
		// for the final composite
		mColor = color.getMat().clone();
		cv::blur(mask, mAlpha, cv::Size(params.blurSize, params.blurSize));
	}

	int ImageInpainter::CalcNumberOfLevels(cv::InputArray color)
	{
		auto numLevels = 1;
		auto size = std::min(color.cols(), color.rows());
		while (size >= 5) {
			size /= 2;
			numLevels++;
			if (numLevels >= params.maxLevels) break;
		}

		return numLevels;
	}
	void ImageInpainter::FillInLowerLv(InpaintingLevel & levelUpper, InpaintingLevel & levelLower)
	{
		cv::Mat3b mColorUpsampled;
		cv::resize(*(levelUpper.GetColorPtr()), mColorUpsampled, levelLower.GetColorPtr()->size(), 0.0, 0.0, cv::INTER_LINEAR);
		cv::Mat2i mPosMapUpsampled;
		cv::resize(*(levelUpper.GetPosMapPtr()), mPosMapUpsampled, levelLower.GetPosMapPtr()->size(), 0.0, 0.0, cv::INTER_NEAREST);
		for (int r = 0; r < mPosMapUpsampled.rows; ++r)
		{
			auto ptr = mPosMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < mPosMapUpsampled.cols; ++c) ptr[c] = ptr[c] * 2 + cv::Vec2i(r % 2, c % 2);
		}

		auto mColorLw = *(levelLower.GetColorPtr());
		auto mMaskLw = *(levelLower.GetMaskPtr());
		auto mPosMapLw = *(levelLower.GetPosMapPtr());

		const int wLw = levelLower.GetColorPtr()->cols;
		const int hLw = levelLower.GetColorPtr()->rows;
		for (int r = 0; r < hLw; ++r)
		{
			auto ptrColorLw = mColorLw.ptr<cv::Vec3b>(r);
			auto ptrColorUpsampled = mColorUpsampled.ptr<cv::Vec3b>(r);
			auto ptrMaskLw = mMaskLw.ptr<uchar>(r);
			auto ptrPosMapLw = mPosMapLw.ptr<cv::Vec2i>(r);
			auto ptrPosMapUpsampled = mPosMapUpsampled.ptr<cv::Vec2i>(r);
			for (int c = 0; c < wLw; ++c)
			{
				if (ptrMaskLw[c] == 0)
				{
					ptrColorLw[c] = ptrColorUpsampled[c];
					ptrPosMapLw[c] = ptrPosMapUpsampled[c];
				}
			}
		}
	}
	void ImageInpainter::BlendBorder(cv::OutputArray dst)
	{
		cv::Mat3f mColorF, mPMColorF, mDstF(levels[0].GetColorPtr()->size());
		mColor.convertTo(mColorF, CV_32FC3, 1.0 / 255.0);
		levels[0].GetColorPtr()->convertTo(mPMColorF, CV_32FC3, 1.0 / 255.0);

		cv::Mat1f mAlphaF;
		mAlpha.convertTo(mAlphaF, CV_32F, 1.0 / 255.0);

		for (int r = 0; r < mColor.rows; ++r)
		{
			auto ptrSrc = mColorF.ptr<cv::Vec3f>(r);
			auto ptrPM = mPMColorF.ptr<cv::Vec3f>(r);
			auto ptrDst = mDstF.ptr<cv::Vec3f>(r);
			auto ptrAlpha = mAlphaF.ptr<float>(r);
			for (int c = 0; c < mColor.cols; ++c)
			{
				ptrDst[c] = ptrAlpha[c] * ptrSrc[c] + (1.0f - ptrAlpha[c]) * ptrPM[c];
			}
		}

		mDstF.convertTo(dst, CV_8UC3, 255.0);
	}
}
