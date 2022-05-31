#include "../include/Directory.hpp"
#include <windows.h>

std::string pathToExeCache;

namespace util
{
	std::string GetExeDirectory()
	{
		if (!pathToExeCache.empty()) return pathToExeCache;
		char pathToExeRaw[MAX_PATH];
		GetModuleFileNameA(NULL, pathToExeRaw, MAX_PATH);
		std::string pathToExe = pathToExeRaw;
		pathToExeCache = pathToExe.substr(0, pathToExe.find_last_of("\\") + 1);
		return pathToExeCache;
	}
}
