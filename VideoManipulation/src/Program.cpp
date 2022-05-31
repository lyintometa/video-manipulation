#include "../include/VideoManipulator.hpp"
#include "../src/Utilities.hpp"

#include <iostream>
#include <chrono>

const std::string templateName("template_50.jpg");

//const std::string pathToSrc("./data/speed_20.jpg");
//const std::string pathToSrc("./data/no_parking.jpg");
//const std::string pathToSrc("./data/speed_20_vid.mp4");

const std::string fileName("speed_20.mp4");

vm::ManipulationParams GetDefaultParameters() {
	vm::ManipulationParams params;

	params.srcPath = util::GetFullNameSource(fileName);
	params.dimensions = cv::Size(1280, 720);
	//params.templateSrcPath = util::GetFullNameSource(templateName);

	return params;
}

int main(int argc, char * argv[])
{
	vm::VideoManipulator manipulator;

	if (argc == 1)
	{
		auto params = GetDefaultParameters();
		manipulator.Init(params);
	}
	else
	{
		manipulator.Init(argc, argv);
	}

	manipulator.Run();
	cv::waitKey();
	return 0;
}