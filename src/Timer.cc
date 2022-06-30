/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include "rumd/Timer.h"

// see: gettimeofday(2), timeradd(3), time(7)

// create and start timer
Timer::Timer():
    t_start(timeval())
{
    gettimeofday(&t_start,0);
}
  
// return elapsed wall time in seconds since timer was created
double Timer::elapsed()
{
    struct timeval t_end;
    gettimeofday(&t_end,0);
    struct timeval t_diff;
    timersub(&t_end, &t_start, &t_diff);
    return (double)t_diff.tv_sec + 1e-6*(double)t_diff.tv_usec;
}
