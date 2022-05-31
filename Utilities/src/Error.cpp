# include "../include/Error.hpp"
#include <string>

namespace err
{
	void Exit()
	{
		std::cout << "Exiting..." << std::endl;
		getchar();
		exit(-1);
	}

	void Exit(std::string msg)
	{
		std::cout << msg << std::endl;
		Exit();
	}
}
