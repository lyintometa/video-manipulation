#pragma once

#include "InpaintingLevel.hpp"

namespace inpainting
{
	InpaintingLevel::InpaintingLevel()
		: borderSize(2), borderSizePosMap(1), windowSize(5), toLeft(0, -1), toRight(0, 1), toUp(-1, 0), toDown(1, 0) 
	{
		vSptAdj =
		{
			cv::Vec2i(-1, -1),	cv::Vec2i(-1, 0),	cv::Vec2i(-1, 1),
			cv::Vec2i(0, -1),						cv::Vec2i(0, 1),
			cv::Vec2i(1, -1),	cv::Vec2i(1, 0),	cv::Vec2i(1, 1)
		};
	}

	InpaintingLevel::~InpaintingLevel() { }

	void InpaintingLevel::Init(const cv::Size initSize, const InpaintingParams& parameters)
	{
		firstFrame = true;
		size = initSize;
		params = parameters;
		std::random_device rnd;
		mt = std::mt19937(rnd());
		cRand = std::uniform_int_distribution<int>(0, size.width - 1);
		rRand = std::uniform_int_distribution<int>(0, size.height - 1);
	}

	void InpaintingLevel::LoadFrame(const cv::Mat3b & color, const cv::Mat1b & mask)
	{
		CreateBorderMat(color, mColor, borderSize);
		CreateBorderMat(mask, mMask, borderSize);

		if (firstFrame) mPosMap[WO_BORDER] = cv::Mat2i(mColor[WO_BORDER].size());
		for (int r = 0; r < mPosMap[WO_BORDER].rows; ++r)
		{
			for (int c = 0; c < mPosMap[WO_BORDER].cols; ++c)
			{
				if (!firstFrame) {
					if (mMask[WO_BORDER](r, c) == 0 && prevMask[WO_BORDER](r, c) != 0) {
						mPosMap[WO_BORDER](r, c) = GetValidRandPos();
					}
					else if (mMask[WO_BORDER](r, c) != 0 && prevMask[WO_BORDER](r, c) == 0) {
						mPosMap[WO_BORDER](r, c) = cv::Vec2i(r, c);
					}
					continue;
				}

				if (mMask[WO_BORDER](r, c) == 0) mPosMap[WO_BORDER](r, c) = GetValidRandPos();
				else mPosMap[WO_BORDER](r, c) = cv::Vec2i(r, c);
			}
		}

		CreateBorderMat(mPosMap[WO_BORDER], mPosMap, borderSizePosMap);
	}

	void InpaintingLevel::Run()
	{
		const float thDist = std::pow(std::max(mColor[WO_BORDER].cols, mColor[WO_BORDER].rows) * params.threshDist, 2.0f);

		for (int i = 0; i < params.maxItr; ++i)
		{
			FwdUpdate(thDist);
			BwdUpdate(thDist);
			Inpaint();
		}

		prevMask[WO_BORDER] = mMask[WO_BORDER].clone();
		prevColor[W_BORDER] = mColor[W_BORDER].clone();
		firstFrame = false;
	}

	cv::Mat3b * InpaintingLevel::GetColorPtr()
	{
		return &(mColor[WO_BORDER]);
	}

	cv::Mat1b * InpaintingLevel::GetMaskPtr()
	{
		return &(mMask[WO_BORDER]);
	}

	cv::Mat2i * InpaintingLevel::GetPosMapPtr()
	{
		return &(mPosMap[WO_BORDER]);
	}

	cv::Size InpaintingLevel::getSize()
	{
		return size;
	}

	void InpaintingLevel::Inpaint()
	{
		for (int r = 0; r < mColor[WO_BORDER].rows; ++r)
		{
			auto ptrColor = mColor[WO_BORDER].ptr<cv::Vec3b>(r);
			auto ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
			for (int c = 0; c < mColor[WO_BORDER].cols; ++c)
			{
				ptrColor[c] = mColor[WO_BORDER](ptrPosMap[c]);
			}
		}
	}

	cv::Vec2i InpaintingLevel::GetValidRandPos()
	{
		cv::Vec2i p;
		do {
			p = cv::Vec2i(rRand(mt), cRand(mt));
		} while (mMask[WO_BORDER](p) != 255);

		return p;
	}

	void InpaintingLevel::CreateBorderMat(cv::InputArray src, cv::Mat* arr, int borderSize)
	{
		cv::copyMakeBorder(src, arr[W_BORDER], borderSize, borderSize, borderSize, borderSize, cv::BORDER_REFLECT); 
		arr[WO_BORDER] = cv::Mat(arr[W_BORDER], cv::Rect(borderSize, borderSize, src.cols(), src.rows()));
	}

	void InpaintingLevel::FwdUpdate(const float thDist)
	{
		const auto scAlpha = params.alpha;
		const auto acAlpha = 1.0f - params.alpha;
		const auto excBeta = params.beta;
		const auto sacBeta = 1.0f - params.beta;

#pragma omp parallel for // NOTE: This is not thread-safe
		for (int r = 0; r < mColor[WO_BORDER].rows; ++r)
		{
			auto ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
			auto ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
			for (int c = 0; c < mColor[WO_BORDER].cols; ++c)
			{
				if (ptrMask[c] != 0) continue;
				cv::Vec2i target(r, c);
				cv::Vec2i ref = ptrPosMap[target[1]];
				cv::Vec2i top = target + toUp;
				cv::Vec2i left = target + toLeft;
				if (top[0] < 0) top[0] = 0;
				if (left[1] < 0) left[1] = 0;
				cv::Vec2i topRef = mPosMap[WO_BORDER](top) + toDown;
				cv::Vec2i leftRef = mPosMap[WO_BORDER](left) + toRight;
				if (topRef[0] >= mColor[WO_BORDER].rows) topRef[0] = mPosMap[WO_BORDER](top)[0];
				if (leftRef[1] >= mColor[WO_BORDER].cols) leftRef[1] = mPosMap[WO_BORDER](left)[1];

				// propagate
				float cost = scAlpha * CalcSptCost(target, ref, thDist) + acAlpha * CalcAppCost(target, ref);
				if (!firstFrame) cost = sacBeta * cost + excBeta * CalcExtAppCost(target, ref);

				float costTop = FLT_MAX, costLeft = FLT_MAX;

				if (mMask[WO_BORDER](top) == 0 && mMask[WO_BORDER](topRef) != 0)
				{
					costTop = scAlpha * CalcSptCost(target, topRef, thDist) + acAlpha * CalcAppCost(target, topRef);
					if (!firstFrame) costTop = sacBeta * costTop + excBeta * CalcExtAppCost(target, topRef);
				}
				if (mMask[WO_BORDER](left) == 0 && mMask[WO_BORDER](leftRef) != 0)
				{
					costLeft = scAlpha * CalcSptCost(target, leftRef, thDist) + acAlpha * CalcAppCost(target, leftRef);
					if (!firstFrame) costLeft = sacBeta * costLeft + excBeta * CalcExtAppCost(target, leftRef);
				}

				if (costTop < cost && costTop < costLeft)
				{
					cost = costTop;
					ptrPosMap[target[1]] = topRef;
				}
				else if (costLeft < cost)
				{
					cost = costLeft;
					ptrPosMap[target[1]] = leftRef;
				}

				// random search
				int itrNum = 0;
				cv::Vec2i refRand;
				float costRand = FLT_MAX;
				do {
					refRand = GetValidRandPos();
					costRand = scAlpha * CalcSptCost(target, refRand, thDist) + acAlpha * CalcAppCost(target, refRand);
					if (!firstFrame) costRand = sacBeta * costRand + excBeta * CalcExtAppCost(target, refRand);
					//if (!reloadPosMap) costRand /= 1 - params.beta;
				} while (costRand >= cost && ++itrNum < params.maxRandSearchItr);

				if (costRand < cost) ptrPosMap[target[1]] = refRand;
			}
		}
	}

	void InpaintingLevel::BwdUpdate(const float thDist)
	{
		const auto scAlpha = params.alpha;
		const auto acAlpha = 1.0f - params.alpha;
		const auto excBeta = params.beta;
		const auto sacBeta = 1.0f - params.beta;

#pragma omp parallel for // NOTE: This is not thread-safe
		for (int r = mColor[WO_BORDER].rows - 1; r >= 0; --r)
		{
			auto ptrMask = mMask[WO_BORDER].ptr<uchar>(r);
			auto ptrPosMap = mPosMap[WO_BORDER].ptr<cv::Vec2i>(r);
			for (int c = mColor[WO_BORDER].cols - 1; c >= 0; --c)
			{
				if (ptrMask[c] == 0)
				{
					cv::Vec2i target(r, c);
					cv::Vec2i ref = ptrPosMap[target[1]];
					cv::Vec2i bottom = target + toDown;
					cv::Vec2i right = target + toRight;
					if (bottom[0] >= mColor[WO_BORDER].rows) bottom[0] = target[0];
					if (right[1] >= mColor[WO_BORDER].cols) right[1] = target[1];
					cv::Vec2i bottomRef = mPosMap[WO_BORDER](bottom) + toUp;
					cv::Vec2i rightRef = mPosMap[WO_BORDER](right) + toLeft;
					if (bottomRef[0] < 0) bottomRef[0] = 0;
					if (rightRef[1] < 0) rightRef[1] = 0;

					// propagate
					float cost = scAlpha * CalcSptCost(target, ref, thDist) + acAlpha * CalcAppCost(target, ref);
					if (!firstFrame) cost = sacBeta * cost + excBeta * CalcExtAppCost(target, ref);

					float costDown = FLT_MAX, costRight = FLT_MAX;

					if (mMask[WO_BORDER](bottom) == 0 && mMask[WO_BORDER](bottomRef) != 0)
					{
						costDown = scAlpha * CalcSptCost(target, bottomRef, thDist) + acAlpha * CalcAppCost(target, bottomRef);
						if (!firstFrame) costDown = sacBeta * costDown + excBeta * CalcExtAppCost(target, bottomRef);
					}
					if (mMask[WO_BORDER](right) == 0 && mMask[WO_BORDER](rightRef) != 0)
					{
						costRight = scAlpha * CalcSptCost(target, rightRef, thDist) + acAlpha * CalcAppCost(target, rightRef);
						if (!firstFrame) costRight = sacBeta * costRight + excBeta * CalcExtAppCost(target, rightRef);
					}

					if (costDown < cost && costDown < costRight)
					{
						cost = costDown;
						ptrPosMap[target[1]] = bottomRef;
					}
					else if (costRight < cost)
					{
						cost = costRight;
						ptrPosMap[target[1]] = rightRef;
					}

					// random search
					int itrNum = 0;
					cv::Vec2i refRand;
					float costRand = FLT_MAX;
					do {
						refRand = GetValidRandPos();
						costRand = scAlpha * CalcSptCost(target, refRand, thDist) + acAlpha * CalcAppCost(target, refRand);
						if (!firstFrame) costRand = sacBeta * costRand + excBeta * CalcExtAppCost(target, refRand);
					} while (costRand >= cost && ++itrNum < params.maxRandSearchItr);

					if (costRand < cost) ptrPosMap[target[1]] = refRand;
				}
			}
		}
	}

	float InpaintingLevel::CalcSptCost(const cv::Vec2i & target, const cv::Vec2i & ref, float maxDist, float w)
	{
		const float normFactor = maxDist * 2.0f;

		float sc = 0.0f;
		for (const auto& v : vSptAdj)
		{
			cv::Vec2f diff((ref + v) - mPosMap[W_BORDER](target + cv::Vec2i(borderSizePosMap, borderSizePosMap) + v));
			sc += std::min(diff.dot(diff), maxDist);
		}

		return sc * w / normFactor;
	}

	float InpaintingLevel::CalcAppCost(const cv::Vec2i & target, const cv::Vec2i & ref, float w)
	{
		const float normFctor = 255.0f * 255.0f * 3.0f;

		float ac = 0.0f;
		for (int r = 0; r < windowSize; ++r)
		{
			uchar* ptrMask = mMask[W_BORDER].ptr<uchar>(r + ref[0]);
			cv::Vec3b* ptrTargetColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + target[0]);
			cv::Vec3b* ptrRefColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + ref[0]);
			for (int c = 0; c < windowSize; ++c)
			{
				if (ptrMask[c + ref[1]] == 0)
				{
					ac += FLT_MAX / 25.0f;
				}
				else
				{
					cv::Vec3f diff(cv::Vec3f(ptrTargetColor[c + target[1]]) - cv::Vec3f(ptrRefColor[c + ref[1]]));
					ac += diff.dot(diff);
				}
			}
		}

		return ac * w / normFctor;
	}

	float InpaintingLevel::CalcExtAppCost(const cv::Vec2i & target, const cv::Vec2i & ref, float w)
	{
		const float normFctor = 255.0f * 255.0f * 3.0f;

		float ac = 0.0f;
		for (int r = 0; r < windowSize; ++r)
		{
			uchar* ptrMask = mMask[W_BORDER].ptr<uchar>(r + ref[0]);
			cv::Vec3b* ptrTargetColor = mColor[W_BORDER].ptr<cv::Vec3b>(r + target[0]);
			cv::Vec3b* ptrRefColor = prevColor[W_BORDER].ptr<cv::Vec3b>(r + ref[0]);
			for (int c = 0; c < windowSize; ++c)
			{
				if (ptrMask[c + ref[1]] == 0)
				{
					ac += FLT_MAX / 25.0f;
				}
				else
				{
					cv::Vec3f diff(cv::Vec3f(ptrTargetColor[c + target[1]]) - cv::Vec3f(ptrRefColor[c + ref[1]]));
					ac += diff.dot(diff);
				}
			}
		}

		return ac * w / normFctor;
	}
}
