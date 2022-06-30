"""Module providing class Autotune for optimizing
the technical parameters of in a rumd simulation """

import time
import getpass # to get the username; some say use os and pwd modules instead
import math
import rumd


def SetSkin(sim, skin):
    """ Set the skin parameter on all potentials associated with sim """
    N = sim.GetNumberOfParticles()
    vol = sim.GetVolume()
    rho = N / vol
    sizeofint = 4 # (on the device)
    for pot in sim.potentialList:
        try:
            cut_sk = pot.GetMaxCutoff() + skin
        except AttributeError:
            continue
        est_num_nbrs = 1.5*4.*math.pi/3*rho*cut_sk**3
        total_num_nbrs = N *est_num_nbrs
        mem_nbrs = total_num_nbrs * sizeofint
        if mem_nbrs > rumd.Device.GetDevice().GetDeviceMemory()//2:
            print("Not enough memory (skin %.2f)" % skin)
            raise MemoryError
        pot.SetNbListSkin(skin)


def SetNB_Method(sim, NB_method):
    """ Set the neighbor-list method on all potentials attached to sim"""
    if NB_method == "sort":
        sim.SetSortInterval(0)

    for pot in sim.potentialList:
        try:
            pot.SetNB_Method(NB_method)
        except:
            pass
            
def Attempt_SetPB_TP(sim, pb, tp, verbose=False):
    """Set pb and tp, call CalcF to check that sufficient resources are
    available. If not set back to old values. Return success or failure after
    calling reset NeighborList to ensure that the first time step will include
    a neighbor-list rebuild """
    pb_before = sim.sample.GetParticlesPerBlock()
    tp_before = sim.sample.GetThreadsPerParticle()

    if verbose:
        print("Setting pb, tp", pb, tp)

    try:
        sim.sample.SetCheckCudaErrors(True)
        sim.SetPB_TP(pb, tp)
        sim.sample.CalcF()
    except ValueError as v:

        if (v.args[0].startswith("Too many threads") or
                v.args[0].startswith("tp too high")):
            print(v.args[0])
        else:
            print("Exception after trying to set pb, tp")
        sim.SetPB_TP(pb_before, tp_before)
        success = False
    else:
        success = True
    finally:
        sim.sample.SetCheckCudaErrors(False)

    for pot in sim.potentialList:
        pot.ResetInternalData()

    return success



def GetMATS(sim, millisec_per_step):
    """ Convert time per step in milliseconds into millions of atom timesteps
    per second (MATS)"""
    return sim.sample.GetNumberOfParticles()*1.e-3/millisec_per_step


class Autotune(object):
    """
    Allows tuning of internal parameters in a Simulation. Typical use:
    [create simulation object, attach potential(s), integrator etc]
    at = Autotune()
    at.Tune(sim)

    """
    def __init__(self, **kwargs):
        """
        keyword arguments such as pb=32 or tp="default" may be specified,
        in order to fix a parameter at a given value or at its default
        value. The parameters with their allowed ranges where applicable are:

        pb [16, 32, 48, 64, 96, 128, 192]
        tp [1, ..., 65]
        skin [non-negative float]
        NB_method ["none", "n2", "sort"]

        The autotuner restricts the ranges depending on system size, for example
        NB_method "none" will not be attempted for N > self.size_stop_no_nb
        which is 5000 by default.

        For the non-sorting-based neighbor-list methods ("none" and "n2"),
        external sorting, at a fixed interval, is possible according to spatial
        position in 1, 2 or 3 dimensions (this is a holdover from the older
        version of rumd but can help a little near the transition from "n2" to
        "sort". The relevant parameters are:

        sorting_scheme ["X", "XY", "XYZ"]
        sort_interval [non-negative integer]

        The keyword argument ignore_file=True forces tuning even if a
        matching autotune.dat file exists.
        """
        self.file_format = 2
        self.auto_filename = "autotune.dat"

        self.test_mode = False
        self.pe_tol = 0.005
        self.pe_ref = None
        self.nSteps_test_mode = 100

        self.nSteps_pre_opt = (10000, 10000)

        self.nSteps1 = (10000, 2000) # phase 1
        self.nSteps2 = (2000, 2000)  # phase 2

        self.nStepsSS = (5000, 5000) # sorting scheme
        self.nStepsSortInterval = (20000, 1000)

        # above this number of particles reduce nsteps
        # proportionately (but keeping above a minimum)
        self.size_reduced_nsteps = 1000

        # above this size do not use NB_method="none"
        self.size_stop_no_nb = 5000
        self.size_stop_n2 = 50000

        self.size_stop_pb_16 = 4000
        self.size_stop_pb_32 = 100000
        self.size_stop_tp_gt_1 = 100000

        self.fraction_cooldown_time = 0.5

        self.test_sort_int = 200

        self.ignore_file = False

        # the device name will be considered a user parameter (ie
        # something given, rather than a technical parameter to tune)
        self.deviceName = rumd.Device.GetDevice().GetDeviceName()

        self.parameter_ranges = {
            "pb": [16, 32, 48, 64, 96, 128, 192],
            "tp": list(range(1, 65)),
            "NB_method":["none", "n2", "sort"],
            "sorting_scheme": ["X", "XY", "XYZ"],
            }

        self.fix_param_values = {}
        for param in ["pb", "tp", "skin",
                      "NB_method", "sorting_scheme", "sort_interval"]:
            self.fix_param_values[param] = None

        # fix parameter values specified as constructor arguments
        for key in kwargs.keys():
            if key == "ignore_file":
                self.ignore_file = kwargs[key]
            elif key  in self.fix_param_values.keys():
                self.fix_param_values[key] = kwargs[key]
            else:
                raise ValueError("Unknown parameter %s" % key)


        self.sortSchemeDict = {"X":rumd.SORT_X,
                               "XY":rumd.SORT_XY,
                               "XYZ":rumd.SORT_XYZ}

        # For converting string representations of technical parameters:
        # int for integers, float for floating point numbers
        # anonymous functions (using the lambda construction) handle
        # lists of integers and lists of floats
        # The second item says whether there will be more than one line in the
        # file with the same key, in which case the items are made into a list
        self.user_prm_convert_func_list = {
            "N":(int, False),
            "rho":(float, False),
            "T":(float, False),
            "potentialType":(None, False),
            "deviceName":(None, False),
            "numberPotentials":(int, False),
            "potentialTypes":(lambda str_list_str: str_list_str.split(","),
                              False),
            "numberEachType":(lambda int_list_str: [int(item)
                                                    for item in int_list_str.split(",")], False),
            "pot_parameters":(lambda float_list_str: [float(item)
                                                      for item in float_list_str.split(",")], True),
            "integratorType":(None, False),
            "timeStep":(float, False)
            }
        # tolerances for matching parameters. None is for strings, which must
        # match exactly.
        self.user_prm_tol = {
            "N": 0,
            "rho":0.05,
            "T": 0.05,
            "numberPotentials": 0,
            "potentialTypes": None,
            "deviceName": None,
            "numberEachType":0,
            "pot_parameters":0.05,
            "integratorType":None,
            "timeStep":0.001}

        self.tech_prm_convert_func = {"skin":float,
                                      "pb":int,
                                      "tp":int,
                                      "NB_method":None,
                                      "sorting_scheme":None,
                                      "sort_interval":int}

        self.current_tech_params = None
        self.moleculeData_pre_opt = None
        self.sample_start = None
        self.moleculeData_start = None
        self.sample_pre_opt = None
        self.itg_infoStr_pre_opt = None
        self.itg_infoStr_start = None
        self.ratio_opt_default = None
        self.sim_verbose = None

    def Tune(self, sim):
        """ Take a Simulation object and run some short runs to determine
        optimal values of the technical parameters pb, tp, skin, etc, saving
        the result in a file. If the file exists and the user parameters are
        the same, then simply use the values from the file
        """
        user_params = self.GetUserParameters(sim)
        self.current_tech_params = self.GetCurrentTechnicalParameters(sim)

        opt_tech_params = None
        if not self.ignore_file:
            opt_tech_params = self.CompareSavedParameters(user_params)

        if opt_tech_params is None:
            opt_tech_params = self.OptimizeTechnicalParameters(sim)
            self.WriteParametersToFile(user_params, opt_tech_params)

        # set sim to have the optimal parameters
        self.SetTechnicalParameters(sim, opt_tech_params)


    def GetUserParameters(self, sim):
        """ Takes a simulation object and returns a dictionary containing the
        user parameters, ie the given details of the simulation that the
        user has chosen to run. Used for deciding whether a given autotune
        file is appropriate for the current simulation.

        For the configuration this means:
        N, ntypes, numberEachType, density, kinetic temperature

        (but not the particle coordinates although those are in principle
        part of the user specification of the simulation)

        The number of potential objects. For the first potential (only the
        first in the list is considered): the
        parameters for each type-pair.

        For the integrator: the type of integrator and the time step
        """

        user_params = {"deviceName":self.deviceName}

        # read the relevant parameters from the simulation
        N = sim.sample.GetNumberOfParticles()
        user_params["N"] = N
        nTypes = sim.sample.GetNumberOfTypes()
        # composition
        numberEachType = []
        for tdx in range(nTypes):
            numberEachType.append(sim.sample.GetNumberThisType(tdx))

        user_params["numberEachType"] = numberEachType
        user_params["rho"] = N/sim.sample.GetSimulationBox().GetVolume()
        user_params["T"] = sim.itg.GetKineticEnergy()/N*2./3.

        user_params["numberPotentials"] = len(sim.potentialList)

        # potential types. Assumes the user doesn't change the ID string
        user_params["potentialTypes"] = [pot_obj.GetID_String() for pot_obj in sim.potentialList]

        # potential parameters (note the numbers here are not necessarily those
        # the user provides but the ones stored internally; they are easier to
        # get from the potential)

        pot_parameters = []
        for pot_obj in sim.potentialList:
            pot_params = []
            try:
                pot_params = pot_obj.GetParameterList(nTypes)
            except:
                pot_params = []
            pot_parameters.append(pot_params)

        user_params["pot_parameters"] = pot_parameters

        itg_Data = sim.itg.GetInfoString(8).split(",")
        user_params["integratorType"] = itg_Data[0]
        user_params["timeStep"] = float(itg_Data[1])

        return user_params

    def GetCurrentTechnicalParameters(self, sim):
        """ Get the technical parameters from sim and return them in a dictionary """
        current_tech_params = {}

        current_tech_params["pb"] = sim.sample.GetParticlesPerBlock()
        current_tech_params["tp"] = sim.sample.GetThreadsPerParticle()
        current_tech_params["skin"] = sim.potentialList[0].GetNbListSkin()
        current_tech_params["NB_method"] = sim.potentialList[0].GetNB_Method()
        current_tech_params["sort_interval"] = sim.runtime_action_intervals["sort"]

        for ss in self.sortSchemeDict.keys():
            if sim.sample.GetSortingScheme() == self.sortSchemeDict[ss]:
                current_tech_params["sorting_scheme"] = ss

        return current_tech_params

    def CompareSavedParameters(self, user_params):
        """ Look for existing autotune file, read the user parameters and
        technical parameters. Then compare the user parameters to the passed
        in ones (the current ones). If they match, return the (previously
        optimized technical parameters.
        """
        # read existing autotune file
        try:
            auto_file = open(self.auto_filename)
        except IOError:
            print("Autotune file not found")
            return None


        # read saved user parameters from autotune file
        saved_user_params = {}
        nextLine = auto_file.readline()
        # find an entry
        while not nextLine.startswith("begin entry"):
            nextLine = auto_file.readline()

        while not nextLine.startswith("autotune version"):
            nextLine = auto_file.readline()

        version = int(nextLine.split()[-1])
        if version != self.file_format:
            print("Version number in autotune file (%d) does not match" \
                  " current version (%d)" % (version, self.file_format))
            return None

        # skip general info and find the user parameters
        while not nextLine.startswith("begin user parameters"):
            nextLine = auto_file.readline()

        nextLine = auto_file.readline()
        while not nextLine.startswith("end user parameters"):
            key, value = nextLine.strip().split("=")
            # value is still encoded as a string at this point
            convert_func, as_list = self.user_prm_convert_func_list[key]

            if convert_func is None:
                # leave as string
                converted_value = value
            else:
                converted_value = convert_func(value)

            # for multiple items with same key (eg potential parameters, one
            # list for each potential object)
            if as_list is False:
                saved_user_params[key] = converted_value
            else:
                if key not in saved_user_params:
                    saved_user_params[key] = []
                saved_user_params[key].append(converted_value)

            nextLine = auto_file.readline()


        nextLine = auto_file.readline()
        while not nextLine.startswith("begin technical parameters"):
            nextLine = auto_file.readline()

        saved_tech_parameters = {}
        nextLine = auto_file.readline()
        while not nextLine.startswith("end technical parameters"):
            key, value = nextLine.strip().split("=")

            convert_func = self.tech_prm_convert_func[key]
            if convert_func is None:
                saved_tech_parameters[key] = value
            else:
                saved_tech_parameters[key] = convert_func(value)
            nextLine = auto_file.readline()

        while not nextLine.startswith("end entry"):
            nextLine = auto_file.readline()

        match = True
        for k in user_params.keys():

            tol = self.user_prm_tol[k]
            # the code for checking agreement within tolerance is a little
            # messy since it covers different types and the possibilities of
            # lists

            # if we don't have a list, then make one containing the single item
            # if we have a list of lists, then flatten it
            if isinstance(user_params[k], list): # a list
                if isinstance(user_params[k][0], list):
                    # list of lists
                    u_items, s_items = [], []
                    for idx in range(len(user_params[k])):
                        u_items += user_params[k][idx]
                        s_items += saved_user_params[k][idx]
                else:
                    u_items, s_items = user_params[k], saved_user_params[k]
            else:
                u_items, s_items = [user_params[k]], [saved_user_params[k]]


            # then compare lists. Break out of the loop when we get False
            for u, s in zip(u_items, s_items):
                # None is for strings
                if (tol is None and u != s) or \
                  (tol is not None and abs(u - s) > tol):
                    match = False
                    break
            if not match:
                break

        if match:
            print("Found matching parameters")
            print(saved_tech_parameters)
            return saved_tech_parameters
        else:
            print("Autotune file found but user parameters do not match")
            return None




    def SetTechnicalParameters(self, sim, tech_params):
        """ Given a sim object and a dictionary of technical parameters,
        set them on sim using the relevant functions"""

        print("Setting optimal technical parameters")
        for key in tech_params.keys():
            print(key, "=", tech_params[key])

        SetSkin(sim, tech_params["skin"])

        sim.SetPB_TP(tech_params["pb"], tech_params["tp"])

        SetNB_Method(sim, tech_params["NB_method"])

        if tech_params["NB_method"] == "n2" and tech_params["sorting_scheme"] != "none":
            sim.sample.SetSortingScheme(self.sortSchemeDict[tech_params["sorting_scheme"]])
            sim.SetSortInterval(tech_params["sort_interval"])

        else:
            sim.SetSortInterval(0)



    def GetParameterRange(self, paramStr, default):
        """ Get the parameter range to actually try for a given
        technical parameter, taking into account whether the user
        decided to keep it fixed at a certain value or at its original
        value """

        if self.fix_param_values[paramStr] is None:
            param_range = self.parameter_ranges[paramStr]
        elif self.fix_param_values[paramStr] == "default":
            param_range = [default]
        else:
            param_range = [self.fix_param_values[paramStr]]

        return param_range

    def GetNumberFixedParams(self):
        """ Tell how many technical parameters are being held fixed """
        num_fixed_prms = 0
        for fp_val in self.fix_param_values.values():
            if fp_val is not None:
                num_fixed_prms += 1
        return num_fixed_prms

    def PreOptimize(self, sim):
        """
        Make copies in order to be able to reset the state after each run
        and do a "pre-optimization" run to reduce transient behavior both
        in the simulation and in the GPU performance.
        """
        self.sample_start = sim.sample.Copy()
        if sim.moleculeData is not None:
            self.moleculeData_start = sim.moleculeData.Copy()
        else:
            self.moleculeData_start = None

        self.itg_infoStr_start = sim.itg.GetInfoString(8)

        nsteps_pre_opt = self.GetNumberOfSteps(sim, self.nSteps_pre_opt)
        if self.test_mode is False and nsteps_pre_opt > 0:
            print("Running %d steps of pre-optimization" % nsteps_pre_opt)
            time_taken = sim.Run(nsteps_pre_opt, suppressAllOutput=True, force_timing=True)
            time.sleep(time_taken/2.) # cooling down

            self.sample_pre_opt = sim.sample.Copy()
            if sim.moleculeData is not None:
                self.moleculeData_pre_opt = sim.moleculeData.Copy()
            else:
                self.moleculeData_pre_opt = None

            self.itg_infoStr_pre_opt = sim.itg.GetInfoString(8)
        else:
            print("No pre-optimization")
            self.sample_pre_opt = self.sample_start
            self.moleculeData_pre_opt = self.moleculeData_start
            self.itg_infoStr_pre_opt = self.itg_infoStr_start


    def GetNumberOfSteps(self, sim, nsteps_small_min):
        """ For small systems, returns the given number of steps. For
        N larger than the internal variable nsteps_small_size the number of
        steps will be scaled accordingly """

        if self.test_mode:
            return self.nSteps_test_mode

        N = sim.GetNumberOfParticles()
        nsteps_small, nsteps_min = nsteps_small_min
        if nsteps_small < nsteps_min:
            raise ValueError("nsteps_small_size must be at least equal to nsteps_min")

        if N < self.size_reduced_nsteps:
            nsteps = nsteps_small
        else:
            nsteps = nsteps_min + (nsteps_small - nsteps_min)*self.size_reduced_nsteps // N

        return nsteps



    def SetFixedParameters(self, sim):
        """ Set parameters to user-specified fixed values, if present """

        if self.fix_param_values["NB_method"] is not None:
            SetNB_Method(sim, self.fix_param_values["NB_method"])
            self.current_tech_params["NB_method"] = self.fix_param_values["NB_method"]
        if self.fix_param_values["skin"] is not None:
            SetSkin(sim, self.fix_param_values["skin"])
            self.current_tech_params["skin"] = self.fix_param_values["skin"]
        if self.fix_param_values["sorting_scheme"] is not None:
            sim.sample.SetSortingScheme(self.fix_param_values["sorting_scheme"])
            self.current_tech_params["sorting_scheme"] = self.fix_param_values["sorting_scheme"]

        if self.fix_param_values["sort_interval"] is not None:
            sim.SetSortInterval(self.fix_param_values["sort_interval"])
            self.current_tech_params["sort_interval"] = self.fix_param_values["sort_interval"]
        pb0 = self.current_tech_params["pb"]
        tp0 = self.current_tech_params["tp"]

        if self.fix_param_values["pb"] is not None:
            pb0 = self.fix_param_values["pb"]
            self.current_tech_params["pb"] = self.fix_param_values["pb"]

        if self.fix_param_values["tp"] is not None:
            tp0 = self.fix_param_values["tp"]
            self.current_tech_params["tp"] = self.fix_param_values["tp"]

        sim.SetPB_TP(pb0, tp0)

    def OptimizeTechnicalParameters(self, sim):
        """ Main function for carrying out optimization """
        # want to time the whole process
        t_start_op = time.time()
        # used for test-mode
        self.pe_ref = None

        # start by setting parameters to user-specified fixed values
        self.SetFixedParameters(sim)

        self.sim_verbose = sim.verbose
        sim.SetVerbose(False)

        sim.write_timing_info = False
        # run some steps to avoid possible transient effects
        self.PreOptimize(sim)

        # get current parameter values
        pb0 = self.current_tech_params["pb"]
        tp0 = self.current_tech_params["tp"]
        skin0 = self.current_tech_params["skin"]
        NB0 = self.current_tech_params["NB_method"]
        ss0 = self.current_tech_params["sorting_scheme"]
        sort_interval0 = self.current_tech_params["sort_interval"]
        if sort_interval0 == 0:
            ss0 = "none"
        opt_tech_params = self.current_tech_params.copy()

        # first do a simulation with default parameters.
        nsteps1 = self.GetNumberOfSteps(sim, self.nSteps1)
        default_time = self.Run(sim, nsteps1)
        print("First, default parameters (or as set by user), " \
              " nsteps=%d" % nsteps1)
        print("NB_method, pb, tp, skin, time (ms/step), MATS")
        print("%s %d %d %.4f %.4f %.3f" % (NB0, pb0, tp0, skin0,
                                           default_time,
                                           GetMATS(sim, default_time)))

        # phase 1: try the different NB_methods, optimizing skin for each one
        if self.fix_param_values["skin"] is not None and \
            self.fix_param_values["skin"] != "default":
            initial_skin = self.fix_param_values["skin"]
            fix_skin = True
        else:
            initial_skin = skin0
            fix_skin = False

        sim.SetSortInterval(0) # only do "external sorting" with none/n2, and first later

        NB_method_list = self.GetParameterRange("NB_method", NB0)
        assert len(NB_method_list) <= len(self.parameter_ranges["NB_method"])
        NB_meth_results = {}
        min_NB_time = 1.e10
        N = sim.GetNumberOfParticles()
        for NB_method in NB_method_list:
            if (NB_method == "n2" and N > self.size_stop_n2) or \
                (NB_method == "none" and N > self.size_stop_no_nb):
                continue

            print("Testing NB_method %s, nsteps=%d" % (NB_method, nsteps1))
            try:
                SetNB_Method(sim, NB_method)
            except RuntimeError as e:
                print("Exception raised when attempting to use NB_method %s. Skipping" % NB_method)
                print("Exception message:", e)
                continue
            del_ln_sk = {True:0.0, False: 0.2}[NB_method == "none" or fix_skin]
            NB_meth_results[NB_method] = self.OptimizeSkin(sim,
                                                           initial_skin,
                                                           del_ln_sk,
                                                           nsteps1,
                                                           verbose=False)

            print("Time %.4f; skin %.4f; MATS %.3f" %
                  (NB_meth_results[NB_method][1],
                   NB_meth_results[NB_method][0],
                   GetMATS(sim, NB_meth_results[NB_method][1])))

            if NB_meth_results[NB_method][1] < min_NB_time:
                min_NB_time = NB_meth_results[NB_method][1]
                min_method = NB_method

        print("Initial result: %s is fastest" % min_method)
        min_time_phase1 = min_NB_time
        # set phase-1 values in case phase 2 fails
        opt_tech_params["skin"] = NB_meth_results[min_method][0]
        opt_tech_params["NB_method"] = min_method


        opt_time = {}
        opt_params2 = {} # for phase 2
        overall_opt_time = 1.0e10
        for NB_method in NB_meth_results:
            # optimize remaining parameters for fastest NB_method(s)
            log_time = NB_meth_results[NB_method][1]
            # all NB-methods giving performance within 20% of fastest
            # are optimized further
            if math.log(log_time) > math.log(min_NB_time) + 0.2 and not self.test_mode:
                continue
            print("Including %s in phase-2 optimization" % NB_method)
            SetNB_Method(sim, NB_method)
            start_skin = NB_meth_results[NB_method][0]
            fix_sk_pb_tp = NB_method == "none" or fix_skin

            opt_sk2, opt_pb2, opt_tp2 = self.OptimizePB_TP(sim, start_skin,
                                                           pb0, tp0, NB_method,
                                                           fix_sk_pb_tp)
            opt_params2[NB_method] = opt_sk2, opt_pb2, opt_tp2
            sim.SetPB_TP(opt_pb2, opt_tp2)
            SetSkin(sim, opt_sk2)

            if NB_method in ["none", "n2"]:
                opt_ss = self.OptimizeSortingScheme(sim, ss0)
                print("Chosen sorting scheme is", opt_ss)
                if opt_ss != "none":
                    sim.sample.SetSortingScheme(self.sortSchemeDict[opt_ss])

                if opt_ss == "none":
                    opt_sort_interval = 0
                    sim.SetSortInterval(0)
                    print("Running %d steps to find time for no (external) sorting" % nsteps1)
                    opt_time[NB_method] = self.Run(sim, nsteps1)
                elif self.fix_param_values["sort_interval"] is None:
                    opt_sort_interval, opt_time[NB_method] = \
                      self.OptimizeSortInterval(sim, start_interval=20)

                else:
                    print("Running to find time for fixed sort-interval")
                    fixed_sort_interval = {True: sort_interval0,
                                           False: self.fix_param_values["sort_interval"]}[self.fix_param_values["sort_interval"] == "default"]
                    sim.SetSortInterval(fixed_sort_interval)
                    opt_time[NB_method] = self.Run(sim, nsteps1)
                    opt_sort_interval = fixed_sort_interval
            else:
                print("Running optimized parameters for this NB_method, %d steps " % nsteps1)
                opt_time[NB_method] = self.Run(sim, nsteps1)

            print("opt_time, %s: %.4f" % (NB_method, opt_time[NB_method]))
            if opt_time[NB_method] < overall_opt_time:
                overall_opt_time = opt_time[NB_method]
                overall_opt_NB_method = NB_method

        # have the overall optimal parameters now

        self.ratio_opt_default = overall_opt_time/default_time
        print("Ratio of time taken for optimized versus default" \
              " parameters %.4f " % self.ratio_opt_default)

        if self.ratio_opt_default > 1.05:
            if min_time_phase1/default_time > 1.05:
                print("Optimization failed, probably due to frequency" \
                      " switching due to overheating, or perhaps because" \
                      " system is highly non-equilibrium. Reverting to " \
                      "original parameters.")
                opt_tech_params = self.current_tech_params.copy()
            else:
                print("Phase 2 optimization (pb/tp) gives apparently worse" \
                      " results, possibly due to temperature problems. Using" \
                      " results of phase 1.")
                # leave opt_tech_params as is
        else:
            print("Final result: NB method %s is fastest" % overall_opt_NB_method)
            opt_sk, opt_pb, opt_tp = opt_params2[overall_opt_NB_method]
            print("Optimal pb, tp:", opt_pb, opt_tp)
            opt_tech_params["skin"] = opt_sk
            opt_tech_params["pb"] = opt_pb
            opt_tech_params["tp"] = opt_tp
            opt_tech_params["NB_method"] = overall_opt_NB_method
            if overall_opt_NB_method in ["none", "n2"]:
                opt_tech_params["sorting_scheme"] = opt_ss
                opt_tech_params["sort_interval"] = opt_sort_interval



        sim.SetPB_TP(opt_tech_params["pb"], opt_tech_params["tp"])
        SetSkin(sim, opt_tech_params["skin"])


        # at the end we copy the original particle data
        # and reset the integrator state to what it was before
        sim.sample.Assign(self.sample_start)
        if self.moleculeData_start is not None:
            sim.moleculeData.Assign(self.moleculeData_start)
            sim.RecreateExclusionList()

        sim.itg.InitializeFromInfoString(self.itg_infoStr_start)
        sim.SetVerbose(self.sim_verbose)
        sim.write_timing_info = True
        t_end_op = time.time()
        print("Time taken for optimizing %.2f seconds" % (t_end_op - t_start_op))

        return opt_tech_params


    def OptimizeSortingScheme(self, sim, ss0):
        """Chooses optimal externally triggered sorting scheme
        (when using the n2 neighbor-list method) """

        nsteps_ss = self.GetNumberOfSteps(sim, self.nStepsSS)
        print("Trying different sorting schemes, nSteps=%d" % nsteps_ss)

        # try with no sorting
        sim.SetSortInterval(0)
        wall_time = self.Run(sim, nsteps_ss)
        w_t_min_all = wall_time
        opt_ss = "none"
        MATS = sim.sample.GetNumberOfParticles() * 1.e-3/wall_time
        print("Sort interval: %d" % self.test_sort_int)
        print("%s %d %d %.4f %.4f" % ("none",
                                      sim.sample.GetParticlesPerBlock(),
                                      sim.sample.GetThreadsPerParticle(),
                                      wall_time, MATS))
        sim.SetSortInterval(self.test_sort_int)
        ss_list = self.GetParameterRange("sorting_scheme", ss0)

        for ss in ss_list:
            sim.sample.SetSortingScheme(self.sortSchemeDict[ss])
            wall_time = self.Run(sim, nsteps_ss)
            print("%s %d %d %.4f %.4f" % (ss,
                                          sim.sample.GetParticlesPerBlock(),
                                          sim.sample.GetThreadsPerParticle(),
                                          wall_time, GetMATS(sim, wall_time)))
            if wall_time < w_t_min_all:
                w_t_min_all = wall_time
                opt_ss = ss

        return opt_ss


    def OptimizePB_TP(self, sim, init_skin, pb0, tp0, NB_method, fix_skin):
        """ Loops through  pb and tp to find the optimal combination. For each
        pair of pb, tp values an optimization with respect to skin is done, so
        the result is the optimal combination of pb, tp and skin (for given
        NB_method and/or sorting details) """
        opt_sk = init_skin
        opt_pb, opt_tp = pb0, tp0
        verbose = False
        # get the ranges to test according to whether parameters have been
        # specified or not.
        N = sim.GetNumberOfParticles()
        pb_list = self.GetParameterRange("pb", pb0)
        tp_list = self.GetParameterRange("tp", tp0)
        if N > self.size_stop_tp_gt_1:
            tp_list = [1]

        nSteps_pb_tp = self.GetNumberOfSteps(sim, self.nSteps2)
        del_ln_sk = {True:0.0, False: 0.1}[fix_skin]
        t_min_all = 1.e10
        init_sk = init_skin
        print("Optimize pb, tp, nSteps=%d" % nSteps_pb_tp)
        print("pb, tp, skin, time")
        for pb in pb_list:
            if(N > self.size_stop_pb_32 and pb <= 32) or \
              (N > self.size_stop_pb_16 and pb == 16):
                continue

            t_min_tp = 1.e10
            tp_vals_since_min = 0
            for tp in tp_list:
                if tp == 1 and NB_method == "none":
                    continue

                ok = Attempt_SetPB_TP(sim, pb, tp, verbose)
                if not ok:
                    print("... breaking out of tp loop")
                    break

                if verbose > 1:
                    print("Running OptimizeSkin")
                min_sk, t_min_sk = self.OptimizeSkin(sim, init_sk, del_ln_sk,
                                                     nSteps_pb_tp,
                                                     verbose=verbose)
                init_sk = min_sk # for next iteration

                print("%d %d %.5f %.5f" % (pb, tp, min_sk, t_min_sk))
                if t_min_sk < t_min_tp:
                    t_min_tp = t_min_sk
                    tp_vals_since_min = 0
                    five_pc_slower = False
                    if t_min_sk < t_min_all:
                        t_min_all = t_min_sk
                        opt_sk, opt_pb, opt_tp = min_sk, pb, tp
                else:
                    five_pc_slower = t_min_sk > 1.05*t_min_tp
                    tp_vals_since_min += 1
                if tp_vals_since_min == 3 or five_pc_slower:
                    break
            print("Best time for this pb: %.4f" % t_min_tp)
            if t_min_tp > 1.1 * t_min_all:
                print("Exceeds overall minimum by more than 10%; breaking " \
                      "out of pb loop")
                break
        print("End of OptimizePB_TP, best parameters are " \
              "(sk/pb/tp) %.3f, %d, %d" % (opt_sk, opt_pb, opt_tp))
        return opt_sk, opt_pb, opt_tp



    def OptimizeSortInterval(self, sim, start_interval):
        """ For none and n2 neighbor-list methods, and a given (external)
        sorting scheme, finds the optimal sorting interval, by starting at the
        average number of steps between NB updates and doubling until the
        wall-time starts to increase"""

        print("Finding optimal sort-interval")
        opt_sort_int_w_time = None
        print("sort_interval, time/step (ms)")
        sort_interval = start_interval

        nsteps_SI = self.GetNumberOfSteps(sim, self.nStepsSortInterval)
        while sort_interval < nsteps_SI:
            sim.SetSortInterval(sort_interval)
            wall_time = self.Run(sim, nsteps_SI)
            print("%d %.4f" % (sort_interval, wall_time))
            if (opt_sort_int_w_time is not None and
                    wall_time > opt_sort_int_w_time[1]):
                break # will only get slower from here ...
            else:
                opt_sort_int_w_time = sort_interval, wall_time
                sort_interval *= 2
        # end while loop

        opt_sort_interval, opt_time = opt_sort_int_w_time
        print("optimal sort_interval", opt_sort_interval)
        return opt_sort_interval, opt_time


    def Run(self, sim, nSteps, verbose=False):
        """ Reset the configuration and integrator to the fixed start values,
        and run a simulation for the given number of time-steps, returning the
        time taken """
        sim.sample.Assign(self.sample_pre_opt)
        if self.moleculeData_pre_opt is not None:
            sim.moleculeData.Assign(self.moleculeData_pre_opt)
            sim.RecreateExclusionList()
        sim.itg.InitializeFromInfoString(self.itg_infoStr_pre_opt, False)

        if verbose:
            print("Running %d steps" % nSteps)

        time_taken = sim.Run(nSteps, suppressAllOutput=True, force_timing=True)
        if verbose:
            print("...done. Time: %.4f sec" % time_taken)
        millisec_per_step = (time_taken)/nSteps*1000.
        if not self.test_mode:
            time.sleep(min(self.fraction_cooldown_time*time_taken, 60.0))

        if self.test_mode:
            sim.sample.CalcF()
            pe = sim.sample.GetPotentialEnergy()
            if self.pe_ref is None:
                self.pe_ref = pe
            else:
                if abs(self.pe_ref - pe) > self.pe_tol:
                    print("pe_ref: %.8g pe: %.8g" % (self.pe_ref, pe))
                    raise ValueError("Test-mode, FAIL: Disagreement between pe and ref-pe")


        return millisec_per_step


    def OptimizeSkin(self, sim, init_sk, del_log_sk, nSteps,
                     verbose=False):
        """ Varying the skin holding other parameters fixed """

        sk_fac = math.exp(del_log_sk)

        # test increasing skin
        skin = init_sk
        SetSkin(sim, skin)
        sk_min = init_sk
        last_wall_time = 1.e10
        next_wall_time = self.Run(sim, nSteps)
        skin_data = [(skin, next_wall_time)]
        min_wall_time = next_wall_time

        # going up (increasing skin)
        while next_wall_time < last_wall_time and sk_fac > 1.:
            last_wall_time = next_wall_time
            skin *= sk_fac
            try:
                SetSkin(sim, skin)
            except MemoryError:
                print("Skin too large (not enough memory)")
                break
            next_wall_time = self.Run(sim, nSteps, verbose=verbose > 1)
            skin_data.append((skin, next_wall_time))
            if next_wall_time < min_wall_time:
                min_wall_time = next_wall_time
                sk_min = skin


        # now going down-reset skin to initial value, and next_wall_time to its
        # first value
        last_wall_time = 1.e10
        next_wall_time = skin_data[0][1] # initial value
        skin = init_sk

        while next_wall_time < last_wall_time and sk_fac > 1.:
            last_wall_time = next_wall_time
            skin /= sk_fac
            SetSkin(sim, skin) # don't expect memory error when going down!
            next_wall_time = self.Run(sim, nSteps)
            skin_data.append((skin, next_wall_time))
            if next_wall_time < min_wall_time:
                min_wall_time = next_wall_time
                sk_min = skin

        if verbose:
            print("skin_data")
            for [s, t] in skin_data:
                print("%.4f %.4g" % (s, t))

        return sk_min, min_wall_time



    def WriteParametersToFile(self, user_params, tech_params):
        """Writes the found optimal technical parameters the the autotune file,
        overwriting an existing file"""

        auto_file = open(self.auto_filename, "w")

        auto_file.write("begin entry\n")
        auto_file.write("autotune version = %d\n" %self.file_format)
        auto_file.write("rumd version=%s\n" % rumd.GetVersion())
        auto_file.write("date=%s\n" % time.ctime())
        auto_file.write("user=%s\n" % getpass.getuser())

        auto_file.write("begin user parameters\n")
        auto_file.write("deviceName=%s\n" % user_params["deviceName"])
        auto_file.write("N=%d\n" % user_params["N"])

        nTypes = len(user_params["numberEachType"])
        numType_0 = user_params["numberEachType"][0]
        total = numType_0
        auto_file.write("numberEachType=%d" % numType_0)
        for tdx in range(1, nTypes):
            numThisType = user_params["numberEachType"][tdx]
            total += numThisType
            auto_file.write(",%d" % numThisType)
        auto_file.write("\n")

        if total != user_params["N"]:
            raise ValueError("WriteParametersToFile: Numbers of each " \
                  "type do not sum to N")

        auto_file.write("rho=%.4f\n" % user_params["rho"])
        auto_file.write("T=%.4f\n" % user_params["T"])
        auto_file.write("numberPotentials=%s\n" % user_params["numberPotentials"])
        auto_file.write("potentialTypes=%s\n" % ",".join(user_params["potentialTypes"]))

        for pdx in range(user_params["numberPotentials"]):
            auto_file.write("pot_parameters=%s\n" % \
                            ",".join(["%.4f" % prm for prm in
                                      user_params["pot_parameters"][pdx]]))

        auto_file.write("integratorType=%s\n" % user_params["integratorType"])
        auto_file.write("timeStep=%.4f\n" % user_params["timeStep"])
        auto_file.write("end user parameters\nbegin technical parameters\n")

        auto_file.write("skin=%.3g\n" % tech_params["skin"])
        auto_file.write("pb=%d\n" % tech_params["pb"])
        auto_file.write("tp=%d\n" % tech_params["tp"])
        auto_file.write("NB_method=%s\n" % tech_params["NB_method"])
        if tech_params["NB_method"] == "n2":
            auto_file.write("sorting_scheme=%s\n" %
                            tech_params["sorting_scheme"])
            auto_file.write("sort_interval=%d\n" %
                            tech_params["sort_interval"])

        auto_file.write("end technical parameters\n")
        auto_file.write("end entry\n")
