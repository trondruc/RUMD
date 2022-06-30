#ifndef TIMER_H
#define TIMER_H

/*
    Copyright (C) 2010  Thomas Schrøder

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    LICENSE.txt file for license details.
*/

#include <sys/time.h>

class Timer
{
private:
  struct timeval t_start;

public:
  Timer();
  double elapsed();
};
#endif // TIMER_H
