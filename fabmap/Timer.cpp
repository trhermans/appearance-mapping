#include "Timer.h"


Timer::Timer(void)
{
	m_startTime = 0;
}

Timer::~Timer(void)
{
}

Timer& Timer::getInstance()
{
	static Timer instance;
	return instance;
}

void Timer::start()
{
	m_startTime = clock();
}

double Timer::getRuntime()
{
	clock_t cputime = clock() - m_startTime;
	return cputime / (double)CLOCKS_PER_SEC;
}

std::string Timer::getRuntimeString()
{
	std::stringstream sstr;
	sstr << "[" << getRuntime() << "s] ";

	return sstr.str();
}