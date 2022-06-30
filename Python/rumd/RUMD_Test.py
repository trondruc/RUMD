""" Defines base class for tests based on running a simulation"""

import os
import gzip
import time

import numpy

import rumd

def OpenLastCompleteTraj(directory, specified_baseFilename=None):
    """
    Attempt to open the "LastComplete" file for trajectory files
    in the specified directory. Unless specified_baseFilename is given,
    try using the basename "trajectory" first,
    then "block". Return the open file object and the found basename.
    """
    if specified_baseFilename is None:
        lastCompFile = None
        attemptedFilenames = []
        for basename in ["trajectory", "block"]:
            filename = directory+"/LastComplete_%s.txt" % basename
            try:
                lastCompFile = open(filename)
            except IOError:
                attemptedFilenames.append(filename)
                continue
            else:
                found_baseFilename = basename
                break
        if lastCompFile is None:
            raise IOError("Could not find any of %s" %str(attemptedFilenames))

    else:
        lastCompFile = open(directory+"/LastComplete_%s.txt" % specified_baseFilename)
        found_baseFilename = specified_baseFilename

    return lastCompFile, found_baseFilename




class RUMD_Test(object):
    """Base class for most test-classes, specifically those tests which
    involve running a rumd simulation and comparing output to reference data"""
    def __init__(self):
        self.generate_reference_data = False

        self.trajectory_tol = {
            "positions":1.e-3,
            "velocities":1.e-3,
            "forces":1.e-3,
            "potential energies":1.e-2,
            "virials":2.e-2,
            }

        self.energy_tol = {
            "pe":1.e-4,
            "ke":1.e-4,
            "p":1.e-4,
            "T":1.e-4,
            "Ps":1.e-3,
            "Etot":1.e-4,
            "W":1.e-4,
            }

        self.mean_energy_tol = {
            "pe":0.02,
            "ke":0.02,
            "p":0.05,
            "T":0.02,
            "Ps":0.2,
            "Etot":0.02,
            "W":0.05,
            }

        self.timingDir = "Timing"
        if not os.access(self.timingDir, os.F_OK):
            os.mkdir(self.timingDir)


        # for comparing variances.
        self.tol_var = 0.005
        self.tol_rel_var = 0.2

        self.numParticlesList = [1000, 2048, 8192]

        self.default_pb_tp = (32, 4)
        self.pb_tp_list = [(16, 1), (16, 2), (16, 4), (16, 8),
                           (32, 1), (32, 2), (32, 4), (32, 8)]

        self.deviceName = rumd.Device.GetDevice().GetDeviceName()
        self.outputfile = None
        self.ref_directory = None
        self.out_directory = None

    def GetTimingFilename(self, mode):
        """ Construct filename for timing data according to mode"""
        filename = self.timingDir + "/%s" %(mode[:2])
        if self.deviceName is not None:
            deviceName_underscore = "_".join(self.deviceName.split())
            filename += "_%s" %  deviceName_underscore

        filename += {True:".ref", False:".out"}[self.generate_reference_data]
        return filename

    def GenerateAllReferenceData(self):
        """ Generate reference data for all three modes"""
        self.RunShort(True)
        self.RunPerformance(True)
        self.RunStress(True)

    def RunShort(self, gen_ref=False, loop_pb_tp=False):
        """
        Run a "short" test using the smallest available system, by default only
        with one combination of pb, tp
        """
        self.generate_reference_data = gen_ref

        timing_filename = self.GetTimingFilename("short")
        open(timing_filename, "w").write("# N pb tp total (s) / scaled" \
                                         " (microseconds/particle/timestep)\n")

        N = self.numParticlesList[0]
        if loop_pb_tp and not gen_ref:
            pb_tp_list = self.pb_tp_list
        else:
            pb_tp_list = [self.default_pb_tp]

        numDifferences = 0
        for (pb, tp) in pb_tp_list:
            numDifferences += self.Test("short", N, pb, tp)

        return numDifferences

    def RunPerformance(self, gen_ref=False):
        """
        Run a "performance" test using all system sizes, looping through
        all pb, tp combinations
        """
        self.generate_reference_data = gen_ref

        timing_filename = self.GetTimingFilename("performance")
        open(timing_filename, "w").write("# N pb tp total (s) / scaled" \
                                         " (microseconds/particle/timestep)\n")

        numDifferences = 0
        for N in self.numParticlesList:
            for (pb, tp) in self.pb_tp_list:
                numDifferences += self.Test("performance", N, pb, tp)

        return numDifferences

    def RunStress(self, gen_ref=False):
        """
        Run a "stress" test on all sizes except the largest, using all
        pb, tp combinations
        """
        self.generate_reference_data = gen_ref

        timing_filename = self.GetTimingFilename("stress")
        open(timing_filename, "w").write("# N pb tp total (s) / scaled" \
                                         " (microseconds/particle/timestep)\n")

        if gen_ref:
            pb_tp_list = [self.default_pb_tp]
        else:
            pb_tp_list = self.pb_tp_list


        numDifferences = 0
        for N in self.numParticlesList[:-1]:
            for (pb, tp) in pb_tp_list:
                numDifferences += self.Test("stress", N, pb, tp)

        return numDifferences

    def Test(self, mode, numParticles, pb, tp):
        """ Run a single test, with specified system size, mode, pb, tp"""
        self.out_directory = "TrajectoryFiles_%s" % mode
        self.ref_directory = "TrajectoryRef_%s_N%d" % (mode, numParticles)
        print("mode=%s" % mode)
        print("Running with N=%d, pb=%d, tp=%d" % (numParticles, pb, tp))

        start_time = time.time()
        self.RunSimulation(mode, numParticles, pb, tp)
        time_taken = time.time() - start_time
        print("Time taken %.2f s" % time_taken)
        numDifferences = 0

        if not self.generate_reference_data:
            if mode in ["short", "stress"]:
                self.outputfile = open("diffs_%s_N%d_pb%d_tp%d.log" % \
                                       (mode, numParticles, pb, tp), "w")

                numDifferences += self.DifferencesSingleRun(mode)

                print("See diffs_XXXX files for differences found")
                if numDifferences > 0:
                    self.outputfile.write("Total number of differences is %d\n" % numDifferences)
                else:
                    self.outputfile.write("No differences found\n")


        timing_filename = self.GetTimingFilename(mode)

        open(timing_filename, "a").write("%d %d %d %.2f %.4f\n" % \
                                          (numParticles, pb, tp, time_taken,
                                           1.e6*time_taken/numParticles/
                                           self.numTimeSteps))

        return numDifferences


    def DifferencesSingleRun(self, mode):
        """ Compare output from test to reference data, either direct,
        ie every single item in trajectory and energies, or just
        statistics, based on energies, depending on the mode """
        return {"short":self.CompareOutputDirect,
                "stress":self.CompareOutputStatistics}[mode]()



    def CompareOutputDirect(self):
        """ Return non-zero if differences occur in output files"""
        num_diffs = self.CompareTrajectoryFiles()
        num_diffs += self.CompareEnergiesFiles()
        return num_diffs

    def CompareConfigurations(self, conf_ref, conf_out):
        """ Check for differences between two configurations
        typically test and reference data"""
        md_ref, md_out = conf_ref.metaData, conf_out.metaData
        maxIndex = md_ref.GetLogLin().maxIndex
        assert maxIndex == md_out.GetLogLin().maxIndex
        tol = self.trajectory_tol
        # Get positions and other data and compare
        differenceDict = {}
        pos_ref = conf_ref.GetPositions()
        pos_out = conf_out.GetPositions()

        max_pos_diff = max(abs((pos_ref - pos_out)).flat)
        if max_pos_diff > tol["positions"]:
            differenceDict["positions"] = (max_pos_diff, tol["positions"])

        type_ref = conf_ref.GetTypes()
        type_out = conf_out.GetTypes()
        max_type_diff = max(abs(type_ref-type_out))
        if max_type_diff > 0:
            differenceDict["types"] = (max_type_diff, 0)

        if md_ref.Get("images"):
            im_ref = conf_ref.GetImages()
            im_out = conf_out.GetImages()
            max_image_diff = max(abs((im_ref-im_out)).flat)
            if max_image_diff > 0:
                differenceDict["images"] = (max_image_diff, 0)

        if md_ref.Get("velocities"):
            vel_ref = conf_ref.GetVelocities()
            vel_out = conf_out.GetVelocities()
            max_vel_diff = max(abs((vel_ref - vel_out)).flat)
            if max_vel_diff > tol["velocities"]:
                differenceDict["velocities"] = (max_vel_diff, tol["velocities"])

        if md_ref.Get("forces"):
            for_ref = conf_ref.GetForces()
            for_out = conf_out.GetForces()
            max_for_diff = max(abs((for_ref - for_out)).flat)
            if max_for_diff > tol["forces"]:
                differenceDict["forces"] = (max_for_diff, tol["forces"])

        if md_ref.Get("pot_energies"):
            pe_ref = conf_ref.GetPotentialEnergies()
            pe_out = conf_out.GetPotentialEnergies()
            max_pe_diff = max(abs(pe_ref - pe_out))
            if max_pe_diff > tol["potential energies"]:
                differenceDict["potential energies"] = (max_pe_diff, tol["potential energies"])
        if md_ref.Get("virials"):
            vir_ref = conf_ref.GetVirials()
            vir_out = conf_out.GetVirials()
            max_vir_diff = max(abs(vir_ref - vir_out))
            #ave_vir_diff = numpy.average(abs(vir_ref - vir_out))
            if max_vir_diff > tol["virials"]:
                differenceDict["virials"] = (max_vir_diff, tol["virials"])

        return differenceDict


    def CompareEnergiesFiles(self, baseFilename="energies", lastComplete_suffix=None):
        """Compare energies data fromtest with reference data"""
        if lastComplete_suffix is None:
            lastCompleteFilename = "LastComplete_%s.txt" % baseFilename
        else:
            lastCompleteFilename = "LastComplete_%s.txt" % lastComplete_suffix

        lastEnergyBlock, blockSize = [int(i) for i in
                                      open(self.ref_directory+"/"+
                                           lastCompleteFilename).readline().split()]

        with open(self.out_directory+"/"+lastCompleteFilename) as last:
            output_lastEnergyBlock, output_blockSize = [int(i) for i in
                                                        last.readline()
                                                        .split()]
        if output_lastEnergyBlock != lastEnergyBlock:
            raise ValueError("Number of %s output blocks output does not" \
                             " agree with reference data" % baseFilename)

        # loop over blocks
        numBlocksWithDiffs = 0
        for blockIdx in range(lastEnergyBlock+1):
            # open ref and output files and read the comment lines
            ref_file = gzip.open(self.ref_directory+"/%s%4.4d.dat.gz" % (baseFilename, blockIdx))
            ref_comment_items = ref_file.readline().decode().split()
            out_file = gzip.open(self.out_directory+"/%s%4.4d.dat.gz" % (baseFilename, blockIdx))
            out_comment_items = out_file.readline().decode().split()

            # compare comment lines
            if len(ref_comment_items) != len(out_comment_items):
                self.outputfile.write("Comment lines do not have the same" \
                                      " number of items in block %d" %
                                      blockIdx)
                numBlocksWithDiffs += 1
                break
            nCommentDiff = False
            for ref_item, out_item in zip(ref_comment_items, out_comment_items):
                if out_item != ref_item:
                    if ref_item.startswith("ioformat=") and \
                        out_item.startswith("ioformat=") and \
                        int(ref_item[9:]) < 3 and int(out_item[9:]) < 3:
                        continue
                    nCommentDiff = True
            if nCommentDiff:
                self.outputfile.write("Different comment lines in %s block %d\n" % (baseFilename, blockIdx))
                numBlocksWithDiffs += 1
                break

            # get columns
            colList = None
            for item in ref_comment_items:
                if item.startswith("columns="):
                    colList = item[8:].split(",")
            if colList is None:
                if baseFilename != "energies" and ref_comment_items[0] == "#":
                    # assume the first line is a list of column descriptors
                    colList = ref_comment_items[1:]
                else:
                    raise ValueError("Could not find column-list")
            nCols = len(colList)

            # loop over remaining lines
            next_ref_line = ref_file.readline()
            next_out_line = out_file.readline()
            max_diffs = numpy.zeros(nCols)
            while next_ref_line:
                assert next_out_line

                ref_dataItems = numpy.array([float(item) for item in next_ref_line.split()])
                out_dataItems = numpy.array([float(item) for item in next_out_line.split()])
                diffs = abs(out_dataItems - ref_dataItems)
                for ddx in range(nCols):
                    if diffs[ddx] > max_diffs[ddx]:
                        max_diffs[ddx] = diffs[ddx]

                next_ref_line = ref_file.readline()
                next_out_line = out_file.readline()

            diff_labels = []
            for ddx in range(nCols):
                if colList[ddx] in self.energy_tol:
                    tol = self.energy_tol[colList[ddx]]
                else:
                    tol = 1.e-4

                if max_diffs[ddx] > tol:
                    diff_labels.append([colList[ddx], max_diffs[ddx]])

            if len(diff_labels) > 0:
                self.outputfile.write("Numerical differences in %s block %d:\n" % (baseFilename, blockIdx))
                for item in diff_labels:
                    self.outputfile.write("Max diff %s is %f\n" % (item[0], item[1]))
                numBlocksWithDiffs += 1
            ref_file.close()
            out_file.close()
        return numBlocksWithDiffs

    def CompareTrajectoryFiles(self, specified_baseFilename=None):
        """ Extract configurations from reference data and from test,
        then call CompareConfigurations to look for differences in them"""
        # find number of blocks
        # read in each block file and compare
        lastTrajFile, ref_baseFilename = OpenLastCompleteTraj(
            self.ref_directory, specified_baseFilename)
        lastTrajBlock, blockSize = [int(i) for i in
                                    lastTrajFile.readline().split()]

        output_lastTrajFile, out_baseFilename = OpenLastCompleteTraj(
            self.out_directory, specified_baseFilename)
        output_lastTrajBlock, output_blockSize = [int(i) for i in
                                                  output_lastTrajFile.readline().split()]

        if output_lastTrajBlock != lastTrajBlock:
            raise ValueError("Number of trajectory output blocks does not" \
                             " agree with reference data")

        conf_out = rumd.Tools.Conf()
        conf_ref = rumd.Tools.Conf()
        numDifferentConfigs = 0

        for blockIdx in range(lastTrajBlock+1):
            conf_out.OpenGZ_File(self.out_directory + "/%s%4.4d.xyz.gz" % (
                out_baseFilename, blockIdx))
            conf_ref.OpenGZ_File(self.ref_directory + "/%s%4.4d.xyz.gz" % (
                ref_baseFilename, blockIdx))
            # read a configuration from each block
            conf_out.ReadCurrentFile()
            conf_ref.ReadCurrentFile()
            maxIndex = conf_ref.metaData.GetLogLin().maxIndex

            # loop through configurations in this block and compare
            for index in range(maxIndex+1):
                differences = self.CompareConfigurations(conf_ref, conf_out)
                if len(differences) > 0:
                    numDifferentConfigs += 1
                    self.outputfile.write("Differences detected in %s%4.4d," \
                                          " index %d. Max diff (tol):\n" %
                                          (out_baseFilename, blockIdx, index))
                    for k in differences.keys():
                        self.outputfile.write("%s:  %f (%f)\n" % (
                            k, differences[k][0], differences[k][1]))


                if index < maxIndex:
                    conf_out.ReadCurrentFile()
                    conf_ref.ReadCurrentFile()

            conf_ref.CloseCurrentGZ_File()
            conf_out.CloseCurrentGZ_File()

        return numDifferentConfigs

    def CompareOutputStatistics(self):
        """ Compare statistics of energies data between test and reference
        using rumd_stats (means+variances)"""
        rs_ref = rumd.Tools.rumd_stats()
        rs_ref.SetDirectory(self.ref_directory)
        rs_ref.ComputeStats()
        meanVals_ref = rs_ref.GetMeanVals()
        meanSqVals_ref = rs_ref.GetMeanSqVals()
        mv_keys = meanVals_ref.keys()

        rs_out = rumd.Tools.rumd_stats()
        rs_out.SetDirectory(self.out_directory)
        rs_out.ComputeStats()
        meanVals_out = rs_out.GetMeanVals()
        meanSqVals_out = rs_out.GetMeanSqVals()

        numDifferences = 0

        for k in mv_keys:
            if k not in meanVals_out:
                raise ValueError("key %s not present in output energy stats" % k)
            diff = meanVals_out[k] - meanVals_ref[k]
            var_ref = meanSqVals_ref[k] - meanVals_ref[k]**2
            var_out = meanSqVals_out[k] - meanVals_out[k]**2


            if abs(diff) > self.mean_energy_tol[k]:
                self.outputfile.write("Different mean value for %s: (ref:" \
                                      " %f, output %f difference %.3g)\n" % (
                                          k, meanVals_ref[k], meanVals_out[k],
                                          meanVals_out[k] - meanVals_ref[k]))
                numDifferences += 1

            # use tol_var as the tolerance when the variance is small
            # where small is defined by tol_var itself. Otherwise use
            # tol_rel_var for the relative difference
            if var_ref < self.tol_var:
                differentVariances = (abs(var_out - var_ref) > self.tol_var)
            else:
                differentVariances = (abs(var_out - var_ref)/var_ref > self.tol_rel_var)
            if differentVariances:
                self.outputfile.write("Different variances for %s: (ref:" \
                                      " %f, output %f)\n" % (k, var_ref,
                                                             var_out))
                numDifferences += 1

        # trajectories--compare output of rumd_rdf
        tol_rdf = 0.1
        rdf_ref = rumd.Tools.rumd_rdf()
        rdf_ref.SetDirectory(self.ref_directory)
        rdf_ref.ComputeAll(1000, 10, 1)
        nTypes = rdf_ref.GetNumTypes()

        rdf_out = rumd.Tools.rumd_rdf()
        rdf_out.SetDirectory(self.out_directory)
        rdf_out.ComputeAll(1000, 10, 1)

        if nTypes != rdf_out.GetNumTypes():
            raise ValueError("Number of types does not match in rdf")

        for type1 in range(nTypes):
            for type2 in range(type1, nTypes):
                rdf12_ref = rdf_ref.GetRDFArray(type1, type2)
                rdf12_out = rdf_out.GetRDFArray(type1, type2)
                rdf_diff_meanSq = numpy.average((rdf12_out - rdf12_ref)**2)
                if rdf_diff_meanSq > tol_rdf:
                    self.outputfile.write("Differences found in rdf found" \
                                          " for types %d, %d: rdf_diff_meanSq"\
                                          " = %f\n" % (type1, type2,
                                                       rdf_diff_meanSq))
                    numDifferences += 1

        return numDifferences
