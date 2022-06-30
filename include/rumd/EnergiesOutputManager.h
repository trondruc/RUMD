#ifndef ENERGIESOUTPUTMANAGER_H
#define ENERGIESOUTPUTMANAGER_H

#include <zlib.h>
#include <vector>
#include <map>
#include "rumd/LogLinOutputManager.h"
#include "rumd/EnergiesMetaData.h"
#include "rumd/ExternalCalculator.h"
#include "rumd/rumd_base.h"

class Sample; class Potential; class Integrator;


class EnergiesOutputManager : public LogLinOutputManager
{
public:
  EnergiesOutputManager(Sample *S, const std::string& outputDirectory,
		 const std::string& baseFilename) :
    LogLinOutputManager(S, outputDirectory, baseFilename),
    metaData(),
    energies_ioformat(rumd_ioformat), 
    gz_energyFile(0),
    externalCalculators() {}
  ~EnergiesOutputManager() {if(gz_energyFile) gzclose(gz_energyFile);}


  void Terminate() {
    if(gz_energyFile) gzclose(gz_energyFile);
    gz_energyFile = 0;
  }
  void AddExternalCalculator(ExternalCalculator* calc);
  void RemoveExternalCalculator(ExternalCalculator* calc);
  void SetMetaData(const std::string &key, bool on) {metaData.Set(key, on);}
  void SetMetaData(const std::string &key, int value) {metaData.Set(key, value);}
  void ShowOptions() const {metaData.ShowOptions();}
  void ShowFileStrings() const {metaData.ShowFileStrings();}

  void RegisterIntegrator(Integrator* itg);
  void UnregisterIntegrator(Integrator* itg);
  
  void RegisterPotential(Potential* pot);
  void UnregisterPotential(Potential* pot);
  EnergiesMetaData metaData;

protected:
  void Write();  
private:
  EnergiesOutputManager(const EnergiesOutputManager&); // no default copyconstructor
  EnergiesOutputManager& operator=(const EnergiesOutputManager&); // no default assignment
  void WriteOneLine();
  void CalcEnergies(std::map<std::string, float>& dataValues);

  unsigned int energies_ioformat;
  gzFile gz_energyFile;

  std::vector<ExternalCalculator*> externalCalculators;
};

#endif // ENERGIESOUTPUTMANAGER_H
