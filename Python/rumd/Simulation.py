"""Module containing the Simulation class"""

import time
import weakref
from collections import OrderedDict

from . import rumdswig as rumd
from .pythonOutputManager import pythonOutputManager

class Simulation(object):

    """
    Contains the main objects and provides basic simulation main loop.
    Once created at minimum the potential and integrator must be set
    before Run() can be called.

    """

    def __init__(self, filename, pb=0, tp=0, verbose=True):
        """
        Create the simulation object, and the underlying sample object, after
        reading in a configuration from the named file. Assign default values
        to various simulation- and output-related parameters. Parameters pb
        and tp set the number of particles per block and threads per particle
        respectively; if not present, default values (depending on system size)
        will be chosen. verbose=False suppresses messages to standard output.

        """
        self.sample = rumd.Sample(pb, tp)
        self.sample.SetVerbose(verbose)
        self.sample.ReadConf(filename)

        self.simulationBox = None
        self.thermostat = False
        self.itg = None
        self.potentialList = [] # except molecule potentials
        self.moleculePotentials = {}
        self.output_is_initialized = False
        self.blockSize = None
        self.moleculeData = None
        self.verbose = verbose

        self.outputParameters = {}
        self.outputParameters["energies"] = ("linear", 256)
        self.outputParameters["trajectory"] = ("logarithmic", 1)
        self.allowedManagers = ["energies", "trajectory"]
        self.pyOutputList = []
        self.outputDirectory = self.sample.GetOutputDirectory()

        self.runtime_actions = {}
        # use ordered dict so we can control the order of iteration
        self.runtime_action_intervals = OrderedDict()
        # need weak references to these functions to avoid reference loops.
        # Cannot make a weak reference to a bound method, so need to store
        # separate weak references to the instance object and function object
        self.runtime_actions["momentum_reset"] = (
            weakref.ref(self),
            weakref.ref(self.ResetMomentum.__func__))
        self.runtime_action_intervals["momentum_reset"] = 100

        self.runtime_actions["sort"] = (
            weakref.ref(self.sample),
            weakref.ref(self.sample.SortParticles.__func__))
        self.runtime_action_intervals["sort"] = 0



        self.nominalNumBlocks = 100
        self.exclusionInfo = {"Bond":set(),
                              "Angle":False,
                              "Dihedral":False}

        self.timing_filename = "rumd_timing_info.dat"
        device_info = rumd.Device.GetDevice().GetDeviceReport()
        with open(self.timing_filename, "w") as timing_file:
            timing_file.write("# %s \n" % device_info)
            timing_file.write("# nsteps, time(s), MATS\n")

        self.last_time_stamp = time.time() # rumd.Device.GetDevice().Time() #
        self.elapsed = 0.
        self.write_timing_info = True

    def SetVerbose(self, vb):
        """
        Turn on/off most messages in Simulation and sample objects.
        sim.SetVerbose automatically calls sim.sample.SetVerbose; call
        to the latter explicitly to only affect sample messages.
        """
        self.verbose = vb
        self.sample.SetVerbose(self.verbose)

    def SetPB_TP(self, pb, tp):
        """
        Change pb and tp for the sample. Used mainly by the autotuner.
        Involves copying of data due to array re-allocation.
        """
        self.sample.SetPB_TP(pb, tp)
        if self.moleculeData is not None:
            self.RecreateExclusionList()

    def WritePotentials(self):
        """
        Write a file containing the pair potential
        and pair force for each associated pair potential.
        """
        # what about molecular potentials eg constraints
        for potential in self.potentialList:
            try:
                potential.WritePotentials(self.sample.GetSimulationBox())
            except:
                pass

    def ReadMoleculeData(self, topologyFile):
        """
        Read data describing how particles are connected as molecules
        and create the MoleculeData object.
        """
        self.moleculeData = rumd.MoleculeData(self.sample, topologyFile)
        self.sample.SetMoleculeData(self.moleculeData)


    def RecreateExclusionList(self):
        """
        Recreates the exclusion list via stored exclusion information. Not
        necessary to be called by user.
        """

        for pot_molecular in self.moleculePotentials.values():
            for non_bond_pot in self.potentialList:
                try:
                    pot_molecular.SetExclusions(non_bond_pot)
                except: # not all potentials have neighbor-lists
                    pass

    def ScaleSystem(self, scaleFactor, direction=None, CM=False):
        """
        Scale the simulation box and particle positions by scaleFactor
        in all directions, if direction is ommitted; otherwise only
        in the direction specified by direction.
        For molecular systems, the centers of mass can be scaled
        while keeping intra molecular distances constant.
        """
        if direction is None and not CM:
            self.sample.IsotropicScaleSystem(scaleFactor)
        elif direction is None and CM:
            self.sample.IsotropicScaleSystemCM(scaleFactor)
        elif not CM:
            self.sample.AnisotropicScaleSystem(scaleFactor, direction)
        else:
            raise ValueError("Anisotropic scaling of the molecular center" \
                  " of mass is not implemented yet")

    def SetSimulationBox(self, simBox):
        """
        Set an alternative simulationBox.
        """
        self.simulationBox = simBox # keep it alive by having a reference
        self.sample.SetSimulationBox(simBox)

    def SetPotential(self, potential):
        """
        Set a potential for this system, replacing any previous
        potentials. DEPRECATED - USE AddPotential, if necessary with clear=True
        """
        if len(self.potentialList) > 0:
            print("[Info] Simulation.SetPotential(): previously assigned" \
                  " potentials will be replaced. Use AddPotential to keep" \
                  " more than one potential.")

        if potential.IsMolecularPotential():
            raise RuntimeError("Must use AddPotential for molecular potentials." \
                "SetPotential is still available for non-molecular potentials" \
                "but deprecated in favour of AddPotential")

        self.potentialList = [potential]
        self.sample.SetPotential(potential)
        
    def AddPotential(self, potential, clear=False):
        """
        Add a potential for this system
        (the total force vector and potential will be the sum over all
        added potentials), if clear=True then previous potentials will be removed
        """
        if potential.GetID_String() in self.moleculePotentials:
            raise RuntimeError("Simulation may only have one potential of type %s" % (potential.GetID_String()))

        if clear:
            self.sample.SetPotential(potential)
            self.moleculePotentials = {}
            self.potentialList = [potential]
        else:
            self.sample.AddPotential(potential)
        
        if not potential.IsMolecularPotential(): # non-molecular potentials
            self.potentialList.append(potential)

            for pot_molecular in self.moleculePotentials:
                pot_molecular.SetExclusions(potential)
        
        else: #  molecular potentials
            self.moleculePotentials[potential.GetID_String()] = potential

            for pot_non_bond in self.potentialList:
                potential.SetExclusions(pot_non_bond)



    def SetIntegrator(self, itg):
        """
        Set the integrator for this system, and call its momentum-resetting
        function.
        """
        self.sample.SetIntegrator(itg)
        self.itg = itg

    def WriteConf(self, filename):
        """
        Write the current configuration to a file.
        """
        self.sample.WriteConf(filename)

    def SetBlockSize(self, blockSize):
        """
        Specify the block-size for output (None means automatic). Should be a
        power of 2 for logarithmic scheduling.
        """
        self.blockSize = blockSize

    def SetOutputDirectory(self, outputDirectory):
        """
        Change the directory for output files, default TrajectoryFiles.
        """
        self.outputDirectory = outputDirectory
        self.sample.SetOutputDirectory(outputDirectory)
        # and for any python output managers
        for py_manager in self.pyOutputList:
            py_manager.outputDirectory = outputDirectory
        
    def GetNumberOfParticles(self):
        """
        Return the number of particles in the sample object.
        """
        return self.sample.GetNumberOfParticles()

    def GetVolume(self):
        """
        Return the volume defined by the simulation box.
        """
        return self.sample.GetSimulationBox().GetVolume()


    def SetOutputMetaData(self, manager_name, **kwargs):
        """
        Access to the output manager to control precision and what gets written.

        Examples
        ---------

        sim.SetOutputMetaData("trajectory", precision=6, virials=True)
        sim.SetOutputMetaData("energies", potentialEnergy=False)
        """
        if manager_name not in self.allowedManagers:
            raise ValueError("SetOutputMetaData -- first argument must be " \
                  "one of" + str(self.allowedManagers))

        if manager_name in self.pyOutputList:
            raise ValueError("SetOutputMetaData not implemented for python" \
                  " output managers")

        for key in kwargs:
            self.sample.SetOutputManagerMetaData(manager_name, key, kwargs[key])

    def AddExternalCalculator(self, calc):
        """
        Add data computed by an external calculator class to the energies files.

        Example
        -------

        alt_pot_calc = AlternatePotentialCalculator(...)
        sim.AddExternalCalculator(alt_pot_calc)

        """
        self.sample.AddExternalCalculator(calc)

    def RemoveExternalCalculator(self, calc):
        """
        Remove/disassociate the calculator object calc from the energies output
        manager
        """
        self.sample.RemoveExternalCalculator(calc)


    def AddOutputManager(self, manager_name, manager_obj):
        """
        Add an existing output manager object (typically a C++ output
        manager), specifying a name which will be used to refer to the
        object when calling SetOutputScheduling and SetOutputMetaData.
        The name can be anything, but the convention is that it matches
        the names of the output files associated with this manager.
        """
        if manager_name in self.allowedManagers:
            raise ValueError("Output manager name %s is " \
                  "already in use" % manager_name)
        self.allowedManagers.append(manager_name)
        self.outputParameters[manager_name] = ("logarithmic", 1)
        self.sample.AddOutputManager(manager_name, manager_obj)


    def NewOutputManager(self, manager_name, write=True):
        """
        Create a new (python) output manager, which will write files in the
        output directory starting with manager_name.
        """
        if manager_name in self.allowedManagers:
            raise ValueError("Output manager name %s is" \
                  " already in use" % manager_name)
        if write:
            output_dir = self.outputDirectory
        else:
            output_dir = None
        self.pyOutputList.append(
            pythonOutputManager(manager_name, output_dir))
        self.allowedManagers.append(manager_name)


    def SetOutputScheduling(self, manager_name, schedule, **kwargs):
        """
        Set scheduling information for an output manager.

        manager must be one of the current managers, which include "energies"
        and "trajectory" and whatever other managers have been added.

        schedule must be one of "none", "linear","logarithmic","loglin",
        "limlin"

        extra keyword arguments may/must be supplied where relevant, e.g.
        interval=100 for linear scheduling (required).
        base=10 for logarithmic scheduling (optional, default base is 1)
        base=1, maxInterval=16 for loglin (required)
        interval=5, numitems=100 for limlin (required)
        """
        if manager_name not in self.allowedManagers:
            raise ValueError("SetOutputScheduling -- first argument must" \
                  " be one of " + str(self.allowedManagers))

        param_names = {"none":[],
                       "linear":["interval"],
                       "logarithmic":["base"],
                       "loglin":["base", "maxInterval"],
                       "limlin":["interval", "numitems"]}

        if schedule not in param_names.keys():
            raise ValueError("SetOutputScheduling -- second argument " \
                  "must be " + str(param_names.keys()))

        match_pyManager = [x for x in self.pyOutputList
                           if x.name == manager_name]
        if len(match_pyManager) > 0: # python manager
            match_pyManager[0].SetOutputScheduling(schedule, **kwargs)
        else: # C++ manager
            if schedule == "logarithmic" and "base" not in kwargs:
                kwargs["base"] = 1
            self.outputParameters[manager_name] = tuple(
                [schedule] + [kwargs[name] for name in param_names[schedule]])


    def RegisterCallback(self, manager_name, function, **kwargs):
        """
        Register a data analysis function (which takes a Sample as argument)
        to be called with a specified time-step interval. Callback functions
        that are used to generate output should return a string without newline
        characters.
        """
        if manager_name not in self.allowedManagers[2:]:
            raise ValueError("SetOutputParameters -- first argument must" \
                  " be one of" + str(self.allowedManagers[2:]))
        for pyManager in self.pyOutputList:
            if pyManager.name == manager_name:
                pyManager.AddCallback(function, **kwargs)


    def SetRuntimeAction(self, name, method, interval):
        """
        Specify that a user-supplied bound method, taking no arguments,
        to be called during the main loop (before the force calculation)
        every interval time steps
        """
        self.runtime_actions[name] = (
            weakref.ref(method.__self__), weakref.ref(method.__func__))
        # (use the python3 way instead of older im_self and im_func)
        self.runtime_action_intervals[name] = interval

    def RemoveRuntimeAction(self, name):
        """
        Remove an item from the list of runtime-actions
        """
        if name in self.runtime_actions:
            del self.runtime_actions[name]
            del self.runtime_action_intervals[name]
        else:
            print("[Info] Tried to remove non-existent runtime" \
                  " action %s" % str(name))

    def ApplyRuntimeActions(self, time_step_index):
        """
        Called during main loop. Not to be called by the user.
        """
        for key in self.runtime_action_intervals.keys():
            weak_obj, weak_func = self.runtime_actions[key]
            interval = self.runtime_action_intervals[key]
            func = weak_func()
            obj = weak_obj()
            if interval > 0 and time_step_index % interval == 0:
                # func is not bound to obj, so we pass obj as the first
                # (here only) argument (the corresponding "self")
                func(obj)

    def SetMomentumResetInterval(self, mom_reset_interval):
        """
        Set how many time steps should go between resetting of
        center of mass momentum to zero.
        """
        self.runtime_action_intervals["momentum_reset"] = mom_reset_interval

    def ResetMomentum(self):
        """
        Sets the total momentum to zero via a Galilean velocity transformation
        """
        # It is necessary to make this function in order to be correctly
        # include it in the runtime_actions dictionary (since at construction,
        # before the integrator has been set, self.sample.GetIntegrator()
        # returns None
        self.sample.GetIntegrator().SetMomentumToZero()

    def SetSortInterval(self, sort_interval):
        """
        Set how many time steps should go between sorting of
        particle positions (default value 200). Not necessary to be
        called by user if using autotune.
        """
        self.runtime_action_intervals["sort"] = sort_interval

    def InitializeOutput(self, num_iter):
        """
        Initialize output managers at start of Run(),
        including choosing block size if not set by user.
        Not necessary to be called by the user.
        """
        num_active = 0
        min_blockSize = 1
        for manager_name, outputParameters in self.outputParameters.items():
            # C++ managers
            if outputParameters[0] == "none":
                self.sample.SetOutputManagerActive(manager_name, False)
                continue
            num_active += 1
            self.sample.SetOutputManagerActive(manager_name, True)

            schedule = outputParameters[0]
            if schedule == "linear" or schedule == "limlin":
                interval = outputParameters[1]
                user_maxIndex = -1
                if schedule == "limlin":
                    user_maxIndex = outputParameters[2]-1

                self.sample.SetLogLinParameters(manager_name, interval,
                                                interval, user_maxIndex)
                if interval > min_blockSize:
                    min_blockSize = interval
            elif schedule == "logarithmic":
                self.sample.SetLogLinParameters(manager_name,
                                                base=outputParameters[1],
                                                maxInterval=0)
            elif schedule == "loglin":
                self.sample.SetLogLinParameters(
                    manager_name,
                    base=outputParameters[1],
                    maxInterval=outputParameters[2])
            else:
                raise ValueError("Simulation.InitializeOutput: Unknown schedule %s" % schedule)


        if self.blockSize is None:
            if num_active > 0:
                nominalBlockSize = num_iter//self.nominalNumBlocks//2
                if min_blockSize < nominalBlockSize:
                    min_blockSize = nominalBlockSize
                blockSize = 1024 # default minimum
                while blockSize < min_blockSize:
                    blockSize *= 2
            else:
                blockSize = num_iter//self.nominalNumBlocks

            self.blockSize = blockSize
        numBlocks = num_iter//self.blockSize + 1

        if self.verbose:
            print("Output blockSize=%d, numBlocks=%d" % (self.blockSize,
                                                         numBlocks))
            print("Size of the last incomplete block", num_iter % self.blockSize)

        # If blocksize is not set for python output managers it's the same.
        self.sample.SetOutputBlockSize(self.blockSize)
        for pyManager in self.pyOutputList:
            if pyManager.blockSize is None:
                pyManager.SetBlockSize(self.blockSize)
            pyManager.Initialize() # reset counters after any previous run

        self.output_is_initialized = True

    def CheckSortingClash(self):
        """ Check whether predefined sorting interval has been set
        when sorting based neighbor-list construction is being used"""
        NB_using_sort = False
        for pot in self.potentialList:
            try:
                if pot.GetNB_Method() == "sort":
                    NB_using_sort = True
            except AttributeError:
                continue
        if NB_using_sort and self.runtime_action_intervals["sort"] > 0:
            print("[Warning] sort_interval for ordinary sorting should be" \
                  " zero when using the sort-based NB method. Setting to" \
                  " zero now.")
            self.SetSortInterval(0)


    def NoteTime(self):
        """ Note elapsed time since since last call to this function
        or since creation of simulation object """

        time_stamp = rumd.Device.GetDevice().Time()
        self.elapsed = time_stamp - self.last_time_stamp
        self.last_time_stamp = time_stamp


    def Run(self, num_iter, initializeOutput=True,
            restartBlock=None, suppressAllOutput=False, force_timing=False):
        """
        Run num_iter time steps, by default initializing/resetting output.
        Specify restartBlock to restart a simulation which has been interrupted.
        Set suppressAllOutput to True to avoid any output (for example for
        equilibration).
        """

        if self.itg is None:
            raise ValueError("Integrator has not been set")
        if len(self.potentialList) == 0:
            raise ValueError("Potential has not been set")

        self.CheckSortingClash()

        initialTimeStep = 0
        if not suppressAllOutput:
            if initializeOutput:
                self.InitializeOutput(num_iter)
            else:
                if not self.output_is_initialized:
                    raise ValueError("Cannot have initializeOutput=False " \
                          "if output has not previously been initialized")

            if restartBlock is not None:
                if not initializeOutput:
                    raise ValueError("When restarting, initializeOutput " \
                          "must be True")
                self.sample.ReadRestartConf(restartBlock)
                initialTimeStep = restartBlock*self.blockSize
        # for timing
        timing = (not suppressAllOutput) or force_timing
        if timing:
            self.NoteTime()

        # main loop
        try:
            iterator = xrange(initialTimeStep, num_iter)
        except NameError: # python3
            iterator = range(initialTimeStep, num_iter)
        for i in iterator:

            self.ApplyRuntimeActions(i)

            # Calculate forces
            need_stress = self.itg.RequiresStress()
            self.sample.CalcF(need_stress)

            if not suppressAllOutput:
                # Called here to ensure synchronization of positions and forces
                self.sample.NotifyOutputManagers(i)

                for manager in self.pyOutputList: # python managers for callback
                    if manager.schedule != "none":
                        if manager.nextCalcTimeStep == i:
                            manager.CalcOutput(self.sample)
                        if manager.nextWriteTimeStep == i:
                            manager.WriteOutput(self.sample)
            self.itg.Integrate() # Update positions and velocities
        # end main loop

        if timing:
            self.NoteTime()
            if self.write_timing_info:
                with open(self.timing_filename, "a") as timing_file:
                    nsteps = num_iter-initialTimeStep
                    timing_file.write("%d %.3f %.3f\n" % (
                        nsteps, self.elapsed,
                        self.sample.GetNumberOfParticles()*nsteps/
                        1e6/self.elapsed))

        rumd.Device.GetDevice().CheckErrors()

        # calculate forces one more time so they are consistent with positions
        # if the user will write the configuration to a file afterwards
        self.sample.CalcF()
        
        # empty last buffers with data
        if not suppressAllOutput:
            for manager in self.pyOutputList:
                if manager.output != []:
                    manager.WriteOutput(self.sample)

        return self.elapsed
