

%nodefaultctor SimulationBox; // disable generation of wrapper for default constructor


class SimulationBox {
 public:
  virtual float GetVolume();
  virtual float GetLength(int dir);
  void CopyAnotherBox(SimulationBox* boxToBeCopied);
  std::string GetInfoString(unsigned int precision);
};

%nodefaultctor BaseRectangularSimulationBox; // disable generation of wrapper for default constructor
class BaseRectangularSimulationBox : public SimulationBox {
};

class RectangularSimulationBox : public BaseRectangularSimulationBox {
};

%feature("autodoc","Simulation box which allows simulation of Couette-type shearing involving a relative shift in the x-direction of periodic images separated in the y-direction.\n") LeesEdwardsSimulationBox; 

// this will prevent warnings about keywords and overloaded functions
%feature("kwargs", 0) LeesEdwardsSimulationBox::LeesEdwardsSimulationBox;

class LeesEdwardsSimulationBox : public BaseRectangularSimulationBox {
 public:
  LeesEdwardsSimulationBox();
  LeesEdwardsSimulationBox(SimulationBox* box_to_copy);
  void SetBoxShift( double box_shift);
  double GetBoxShift();
  void IncrementBoxStrain(double inc_boxStrain);
};
