import string

template_code = """
void %(class_name)s::CalcF( bool initialize, bool calc_stresses) {


  neighborList.UpdateNBlist();
  
  char flags =  (calc_stresses << 3) | (cutoffMethod << 1) | initialize;
  switch(flags) {
"""

# we create a single big switch statement
# and loop through all values of the template parameters.
# Combine the values bitwise to make a single value to test
for calc_stresses in ["0","1"]:
    for cutoff_method in ["NS","SP","SF"]:
        for initialize in ["0","1"]:
            template_code += """
  case (%(STR)s<<3 | %(CM)s << 1 | %(INIT)s ):
  if(testLESB) {
    if(kp.threads.y > 1)
      Calcf_NBL_tp<%(STR)s,%(CM)s,%(INIT)s><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
  							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, particleData->d_f, 
							          particleData->d_w, particleData->d_sts, 
							          testLESB, testLESB->GetDevicePointer(),
							          d_params, particleData->GetNumberOfTypes(), 
							          neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
  else
      Calcf_NBL_tp_equal_one<%(STR)s,%(CM)s,%(INIT)s><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
  							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, particleData->d_f, 
							          particleData->d_w, particleData->d_sts, 
							          testLESB, testLESB->GetDevicePointer(),
							          d_params, particleData->GetNumberOfTypes(), 
							          neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());

  }
  else {
    if(kp.threads.y > 1)
      Calcf_NBL_tp<%(STR)s,%(CM)s,%(INIT)s><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, particleData->d_f, 
							          particleData->d_w, particleData->d_sts, 
							          testRSB, testRSB->GetDevicePointer(),
							          d_params, particleData->GetNumberOfTypes(), 
							          neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
    else
      Calcf_NBL_tp_equal_one<%(STR)s,%(CM)s,%(INIT)s><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, particleData->d_f, 
							          particleData->d_w, particleData->d_sts, 
							          testRSB, testRSB->GetDevicePointer(),
							          d_params, particleData->GetNumberOfTypes(), 
							          neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
  }
  break;
""" % {"STR":calc_stresses, "CM":cutoff_method, "INIT":initialize}

template_code += """
  }

}
"""

# similarly for the function CalcF_Local

# we pass a local array for pe and for virial, but the stress array is the one in ParticleData because we don't calcualte stress here so it's safe.
# The result of the virial is for now not used for anything.

template_code += """
void %(class_name)s::CalcF_Local() {
  neighborList.UpdateNBlist();
  
  if(allocated_size_pe != particleData->GetNumberOfVirtualParticles())
  AllocatePE_Array(particleData->GetNumberOfVirtualParticles());

  switch( cutoffMethod ){
"""
for cutoff_method in ["NS","SP","SF"]:
    template_code += """
  case %(CM)s:
    if(testLESB) {
       if(kp.threads.y > 1)
         Calcf_NBL_tp<0,%(CM)s,1><<<kp.grid, kp.threads, shared_size>>>( this,
                                      particleData->GetNumberOfParticles(), 
			 				          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, d_f_pe, d_w_pe, 
                                      particleData->d_sts, 
                                      testLESB, testLESB->GetDevicePointer(),
                                      d_params, particleData->GetNumberOfTypes(), 
                                      neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
      else
        Calcf_NBL_tp_equal_one<0,%(CM)s,1><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
			 				          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, d_f_pe, d_w_pe, 
                                      particleData->d_sts, 
                                      testLESB, testLESB->GetDevicePointer(),
                                      d_params, particleData->GetNumberOfTypes(), 
                                      neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
    }
    else {
      if(kp.threads.y > 1)
        Calcf_NBL_tp<0,%(CM)s,1><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, d_f_pe, d_w_pe,
                                      particleData->d_sts, 
                                      testRSB, testRSB->GetDevicePointer(),
                                      d_params, particleData->GetNumberOfTypes(), 
                                      neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());
      else
           Calcf_NBL_tp_equal_one<0,%(CM)s,1><<<kp.grid, kp.threads, shared_size>>>( this, particleData->GetNumberOfParticles(), 
							          particleData->GetNumberOfVirtualParticles(),
							          particleData->d_r, d_f_pe, d_w_pe, 
                                      particleData->d_sts, 
                                      testRSB, testRSB->GetDevicePointer(),
                                      d_params, particleData->GetNumberOfTypes(), 
                                      neighborList.GetNumNbrsPtr(), neighborList.GetNbListPtr());                                                                  
    }
    break;
    """ % {"CM":cutoff_method}

template_code += """
  }
}
"""

# look in PairPotentials.h to get all the class names

dot_h_file = open("../include/rumd/PairPotential.h")
classNames = []
nextLine = dot_h_file.readline()
while nextLine:
    items = nextLine.split()
    # find lines which begin a new PairPotential class (but not the base class)
    if len(items) > 3 and items[0] == "class":
        clsName = items[1]
        baseName = items[4]
        if clsName.endswith(":"):
            clsName = clsName[:-1]
            baseName = items[3]
        if baseName.endswith("{"):
            baseName = baseName[:-1]
        # checking baseName is to avoid wrapper classes which change the interface a bit but do not have a different CalcF. However it is not guaranteed that this check is sufficient (for example if we derive from an existing PairPotential but still make a new CalcF)
        if clsName != "PairPotential" and baseName == "PairPotential":
            classNames.append(clsName)
    nextLine = dot_h_file.readline()

outfile = open("PairPotentialFunctionBodies.inc","w")
outfile.write("""
///////////////////////////////////////////////////////////////////////////////
// This file has been generated by Generate_PP_FunctionBodies.py, do not edit!
///////////////////////////////////////////////////////////////////////////////
""")

for item in classNames:
    outfile.write( template_code % {"class_name":item} )
outfile.close()
