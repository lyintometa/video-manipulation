#pragma once

#include <iostream>

namespace err 
{
	void Exit();
	void Exit(std::string msg);

	class ParamsException : public std::exception
	{
	public:
		ParamsException() {	}
		ParamsException(std::string paramName, std::string moduleName) {
			msg = "Invalid " + paramName + " provided in " + moduleName + "Params";
		}

		virtual const char * what() const noexcept override {
			return msg.c_str();
		}

	private:
		std::string msg;
	};
}