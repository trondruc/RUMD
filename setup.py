
from distutils.core import setup
from distutils.cmd import Command
#from distutils.command.install import install
from distutils.command.build import build

import sys
import subprocess
import multiprocessing

class rumd_build_cmd(build):
    def run(self):
        # build rumd
        cmd = ['make']
            
        try:
            cmd.append('-j%d' % multiprocessing.cpu_count())
        except NotImplementedError:
            print 'Unable to determine number of CPUs. Using single threaded make.'

        def compile():
            subprocess.call(cmd)

        self.execute(compile, [], "Compiling rumd")
        # I don't see why there's an extra layer of indirection here
        # why not: self.execute(subprcoess.call, [cmd], "Compiling rumd") ??

        # run original build code
        build.run(self)


        

class rumd_test_cmd(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        retcode = subprocess.call(['make', 'test'])
        #raise SystemExit(subprocess.call(['make', 'test')])
                                                    

        
#py_modules=['Autotune', 'pythonOutputManager','RunCompress','Simulation','analyze_energies']
                    
setup(
    name='rumd',
    author='Nicholas Bailey',
    author_email='nbailey@ruc.dk',
    url='http://rumd.org',
    version='3.5',
    description='C++/CUDA-based molecular dynamics simulation code for nVidia GPUs',
    packages=['rumd'],
    package_dir={'':'Python'},

    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                    'Programming Language :: C++',
             'Programming Language :: CUDA',    
'Operating System :: Unix',
                 'Intended Audience :: Science/Research',],
    cmdclass={'build': rumd_build_cmd,
              'test': rumd_test_cmd},
      
      )


# requires ... other packages etc.

# scripts argument (secion 12.5 in distutils docs)...set the first line of scripts to refer to the python that's running setup. Useful for test scripts?
