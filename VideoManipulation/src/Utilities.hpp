#pragma once

#include <iostream>
#include <vector>

namespace util
{
	enum class MediaType
	{
		Undefined,
		Image,
		Video
	};

	std::string GetFullNameSource(std::string fileName = "");
	std::string GetFullNameCopy(std::string fileName = "");
	std::string GetFullNameResult(std::string srcFileName = "");
	std::string GetFullNameDetections(std::string srcFileName = "");
	std::string GetFullNameGeneratedMask(std::string srcFileName = "");
	MediaType GetMediaType(std::string fileName);

	int VectorAverage(std::vector<int>& vec);
}
