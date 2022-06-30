#ifndef STRESSAUTOCORRELATIONFUNCTION_H
#define STRESSAUTOCORRELATIONFUNCTION_H


class PairPotential;
class Sample;

class StressAutocorrelationFunction {
public:
  StressAutocorrelationFunction(int lvec, float kBT, int isample, int printingScheme, int sacfPrintingFrequency=100);
  virtual ~StressAutocorrelationFunction();
  
  void Update(Sample* S);
private:
  int lvec;
  float kBT;
  double isample;
  int sacfPrintingFrequency;
  int printingScheme;

  int index;
  int nsample;
  double *stressOS;
  double *sacf10;
  double *sacf3;
};

#endif // STRESSAUTOCORRELATIONFUNCTION_H
