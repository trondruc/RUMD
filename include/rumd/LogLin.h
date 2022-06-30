#ifndef LOGLIN_H
#define LOGLIN_H


// this structure handles the "logarithmic saving"
class LogLin{
 public:
  unsigned long int block;          // identifies the current block
  unsigned long int nextTimeStep;   // identifies the next time step to write
  unsigned long int base;           // size of smallest interval between writes
  unsigned long int index;          // labels items within a block
  unsigned long int maxIndex;       // value of index for the last item in block
  unsigned long int maxInterval;    // overrides log-indexing to give linear
  unsigned long int increment;      // for internal bookkeeping
  LogLin() : block(0), nextTimeStep(0), base(1), index(0), maxIndex(0), maxInterval(0), increment(0) {}

  // to be able to easily copy a LogLin structure
  constexpr LogLin& operator = (const LogLin& logLin) {
    if(this != &logLin) {
      block = logLin.block;
      nextTimeStep = logLin.nextTimeStep;
      base = logLin.base;
      index = logLin.index;
      maxIndex = logLin.maxIndex;
      maxInterval = logLin.maxInterval;
      increment = logLin.increment;
    }
    return *this;
  }
};

#endif // LOGLIN_H
