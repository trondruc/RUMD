"""Provides class pythonOutputManager for user-defined output managers """

import gzip

class pythonOutputManager(object):
    """ Class for defining new output managers at Python level"""
    def __init__(self, name, outputDirectory="TrajectoryFiles"):
        """ create an output manager. name is used as id and for file names"""
        self.name = name
        self.outputDirectory = outputDirectory
        self.fileHeader = "# "
        self.schedule = None
        self.blockSize = None
        self.base = 1
        self.maxInterval = 0


        self.callbackList = []
        self.output = []
        self.nextCalcTimeStep = 0
        self.nextWriteTimeStep = 0
        self.block = 0
        self.blockNumber = None # is this used for anything? [NB 23/11/18]

    def Initialize(self):
        """ Re-initialize for example to do another run with the same settings"""
        self.output = []
        self.nextCalcTimeStep = 0
        self.nextWriteTimeStep = 0
        self.block = 0
        
        
    def SetOutputScheduling(self, schedule, **kwargs):
        """Set parameters for the output managers.
        Works the same as for the standard (C) output managers, with the
        exception that the blocksize can be different."""
        allowedSchedules = ["none", "linear", "logarithmic", "loglin"]
        if schedule not in allowedSchedules:
            raise ValueError("SetOutputScheduling -- second argument must" \
                             " be " + str(allowedSchedules))
        self.schedule = schedule

        if self.schedule in ("none", "linear"):
            self.base = kwargs["interval"]
        else:
            if "base" in kwargs:
                self.base = kwargs["base"]
        if self.schedule == "loglin":
            self.maxInterval = kwargs["maxInterval"]

        if "blockSize" in kwargs:
            self.blockSize = kwargs["blockSize"]

    def SetBlockSize(self, blockSize):
        """Set the block size of the output manager.
        Can be different for different output managers."""
        self.blockSize = blockSize

    def SetBlockNumber(self, blockNumber):
        """Used by Simulation when continuing a previous simulation."""
        self.blockNumber = blockNumber

    def AddCallback(self, callback, **kwargs):
        """add a callback function to the output manager
        header is an optional argument to pass a string
        to be included in the file header"""
        self.callbackList.append(callback)
        if "header" in kwargs:
            self.fileHeader += kwargs["header"]

    def CalcOutput(self, sample):
        """Execute this manager's callback functions."""
        if self.schedule == "none":
            for callback in self.callbackList:
                callback(sample)
        else:
            line = ""
            for callback in self.callbackList:
                outputStr = callback(sample)
                if outputStr is not None and len(outputStr) > 0:
                    line += outputStr + " "
            if len(line) > 0:
                line += "\n"
            self.output.append(line)

        # calculate the next time step for executing the callback functions
        if self.schedule in ("none", "linear"):
            self.nextCalcTimeStep += self.base
        else:
            blockTimeStep = self.nextCalcTimeStep%self.blockSize
            finishedBlocks = self.block * self.blockSize
            if blockTimeStep == 0:
                self.nextCalcTimeStep += self.base
            elif self.maxInterval == 0 or self.maxInterval > blockTimeStep:
                self.nextCalcTimeStep = finishedBlocks + blockTimeStep * 2
            else:
                self.nextCalcTimeStep += self.maxInterval

    def WriteOutput(self, sample):
        """Write a block of data to a file."""
        if self.outputDirectory is None:
            return
        
        if self.nextWriteTimeStep != 0 and self.output != []:
            fileout = gzip.open(self.outputDirectory + "/%s%04d.dat.gz"
                                %(self.name, self.block), "w")
            fileout.write((self.fileHeader + "\n").encode('utf-8'))
            for line in self.output:
                fileout.write(line.encode('utf-8'))
            fileout.close()

            open(self.outputDirectory+"/LastComplete_%s.txt"
                 %self.name, "w").write("%d %d\n" %(self.block,
                                                    self.blockSize))
            self.output = []
            self.block += 1
            if self.schedule in ("logarithmic", "loglin"):
                self.nextCalcTimeStep = self.nextWriteTimeStep
                self.CalcOutput(sample)


        # calculate the time step for writing the next block
        if self.schedule == "linear":
            self.nextWriteTimeStep = (self.block+1) * self.blockSize -1
        elif self.schedule in ("logarithmic", "loglin"):
            self.nextWriteTimeStep = (self.block+1) * self.blockSize
