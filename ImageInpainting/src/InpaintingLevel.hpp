#pragma once

#include "opencv2/opencv.hpp"
#include <random>

namespace inpainting
{
	struct InpaintingParams
	{
		int maxLevels = 6;			// max number of pyramid levels
		int maxItr = 1;				// max iteration per pyramid level
		int maxRandSearchItr = 1;	// max number of random sampling per pixel
		float alpha = 0.05f;		// balancing parameter between spatial and appearance cost
		float beta = 0.999f;		// balancing parameter between spatial/appearance and temporal cost
		float threshDist = 0.5f;	// 0.5 means the half of the width/height is the maximum
		int blurSize = 5;			// blur kernel size for the final composition
	};

	class InpaintingLevel
	{
	public:
		InpaintingLevel();
		~InpaintingLevel();

		void Init(const cv::Size initSize, const InpaintingParams& parameters);
		void LoadFrame(const cv::Mat3b& color, const cv::Mat1b& mask);
		void Run();

		cv::Mat3b* GetColorPtr();
		cv::Mat1b* GetMaskPtr();
		cv::Mat2i* GetPosMapPtr();

		cv::Size getSize();

	private:
		InpaintingParams params;
		cv::Size size;
		std::mt19937 mt;
		std::uniform_int_distribution<int> cRand;
		std::uniform_int_distribution<int> rRand;
		const int borderSize;
		const int borderSizePosMap;
		const int windowSize;

		enum { WO_BORDER = 0, W_BORDER = 1 };
		cv::Mat3b mColor[2];
		cv::Mat1b mMask[2];
		cv::Mat2i mPosMap[2];

		bool firstFrame;
		cv::Mat1b prevMask[2];
		cv::Mat1i prevSptCostMap[2];
		cv::Mat3b prevColor[2];

		const cv::Vec2i toLeft;
		const cv::Vec2i toRight;
		const cv::Vec2i toUp;
		const cv::Vec2i toDown;
		std::vector<cv::Vec2i> vSptAdj;

		void Inpaint();

		cv::Vec2i GetValidRandPos();
		void CreateBorderMat(cv::InputArray src, cv::Mat* arr, int borderSize);
		void FwdUpdate(const float thDist);
		void BwdUpdate(const float thDist);

		float CalcSptCost(
			const cv::Vec2i& target,
			const cv::Vec2i& ref,
			float maxDist,		// tau_s
			float w = 0.125f	// 1.0f / 8.0f
		);

		float CalcAppCost(
			const cv::Vec2i& target,
			const cv::Vec2i& ref,
			float w = 0.04f		// 1.0f / 25.0f
		);

		float CalcExtAppCost(
			const cv::Vec2i& target,
			const cv::Vec2i& ref,
			float w = 0.04f		// 1.0f / 25.0f
		);
	};
}