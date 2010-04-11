#pragma once

#define TheTimer Timer::getInstance()


#include <string>
#include <sstream>
#include <ctime>


class Timer
{
public:
	Timer(void);
	~Timer(void);

	static Timer& getInstance();

	void start();
	double getRuntime();
	std::string getRuntimeString();

protected:
	clock_t m_startTime;
};
