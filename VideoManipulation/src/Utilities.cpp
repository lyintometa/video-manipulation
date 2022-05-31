#include "Utilities.hpp"
#include "Directory.hpp"
#include <windows.h>
#include <sstream>
#include <numeric>

namespace util
{
	const std::string dirMedia = "Media\\";
	const std::string dirResults = "Results\\";

	const std::string postfixSource = "_src";
	const std::string postfixResult = "_res";
	const std::string postfixDetections = "_det";
	const std::string postfixMask = "_mask";

	std::string dateTime;
	std::string pathToResults;

	std::string GetTime()
	{
		if (!dateTime.empty()) return dateTime;
		SYSTEMTIME st;
		GetSystemTime(&st);
		std::stringstream ss;
		ss << st.wYear << "_" << st.wMonth << "_" << st.wDay << "_";
		ss << st.wHour << "_" << st.wSecond;
		dateTime = ss.str();
		return dateTime;
	}

	std::string GetDirMedia()
	{
		return GetExeDirectory() + dirMedia;
	}

	std::string GetDirResults()
	{
		if (!pathToResults.empty()) return pathToResults;
		pathToResults = GetExeDirectory() + dirResults;
		CreateDirectory(pathToResults.c_str(), NULL);
		return pathToResults;
	}

	std::string ExtractFileName(std::string fileName)
	{
		auto index = fileName.find_last_of("\\");
		if (index == -1) return fileName;
		return fileName.substr(index + 1);
	}

	std::string GetFullName(std::string postfix, std::string fileName = "") {
		if (fileName.empty()) return GetDirResults() + GetTime() + postfix + ".mp4";
		fileName = ExtractFileName(fileName);
		return GetDirResults() + fileName.insert(fileName.find_last_of("."), postfix);
	}

	std::string GetFullNameSource(std::string fileName)
	{
		if (fileName.empty()) GetFullName(postfixSource);
		return GetDirMedia() + fileName;
	}

	std::string GetFullNameCopy(std::string fileName)
	{
		return GetFullName("", fileName);
	}

	std::string GetFullNameResult(std::string fileName)
	{
		return GetFullName(postfixResult, fileName);
	}

	std::string GetFullNameDetections(std::string fileName)
	{
		return GetFullName(postfixDetections, fileName);
	}

	std::string GetFullNameGeneratedMask(std::string fileName)
	{
		return GetFullName(postfixMask, fileName);
	}

	MediaType GetMediaType(std::string fileName)
	{
		auto indexOfFileEnding = fileName.find_last_of(".");
		if (indexOfFileEnding == -1)
		{
			std::cout << "[WARNING] No file ending found in file name '" << fileName.c_str()
				<< "'. Media type could not be determined!"	<< std::endl;
			return MediaType::Undefined;
		}

		auto ending = fileName.substr(indexOfFileEnding);
		if (ending == ".jpg") return MediaType::Image;
		if (ending == ".png") return MediaType::Image;
		if (ending == ".mp4") return MediaType::Video;
		std::cout << "[WARNING]: File ending '" << ending << "' not supported!" << std::endl;
		return MediaType::Undefined;
	}

	int VectorAverage(std::vector<int>& vec)
	{
		return 1.0 * std::accumulate(vec.begin(), vec.end(), 0.0 ) / vec.size();
	}
}