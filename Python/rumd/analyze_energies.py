"""Analysis functions for energies.

This module collects some analysis functions, such as autocorrelation,
crosscorrelation, and functions to perform logarithmic data binning.
The module also defines a class AnalyzeEnergies that can be used for
specifically analyzing the output in the energy files.
"""

import gzip
import math
import numpy as np

def create_logarithmic_bins(min_x, max_x, bins_per_decade=50):
    """Creates a numpy array of bin edges, spaced logarithmically.

    Given a minimum, a maximum, and the number of bins per decade, this
    function returns a numpy array of bin edges.
    """
    log_min_x = math.log10(min_x)
    log_max_x = math.log10(max_x)
    num_bins = (log_max_x - log_min_x) * bins_per_decade
    return np.logspace(log_min_x, log_max_x, int(num_bins + 1))


def data_binning(x, y, x_bins):
    """Returns data x and y, binned in the x direction.

    Given two data sets, both are smoothed by binning the data according
    to the bin edges in x_bins. Returns two numpy arrays of the binned x and y
    data.
    """
    histogram = np.histogram(x, x_bins)[0]
    binned_x = np.histogram(x, x_bins, weights=x)[0]
    binned_y = np.histogram(x, x_bins, weights=y)[0]

    # Remove bins without data points
    nonzero_bins = np.nonzero(histogram)
    histogram = histogram[nonzero_bins]
    binned_x = binned_x[nonzero_bins]
    binned_y = binned_y[nonzero_bins]

    # Normalize x an y values by number of data points in each bin
    binned_x /= histogram
    binned_y /= histogram
    return binned_x, binned_y


def write_columns_to_file(filename, columns, fmt='%g'):
    """Writes a dictionary of columns to a file."""
    header = b"# "
    output = []
    for key in columns:
        header = header + b' ' + str.encode(key)
        output.append(columns[key])
    header = header + b'\n'
    if filename.split('.')[-1] == 'gz':
        f = gzip.open(filename, 'w')
    else:
        f = open(filename, 'w')
    f.write(header)
    np.savetxt(f, np.column_stack(output), fmt=fmt)
    f.close()
    print("wrote file ", filename)


def autocorrelation(y, mean_y=None, length=None):
    """Returns the autocorrelation of an array.

    Make sure that the length of the arrays is a power of two
    if speed is an issue."""
    length_y = len(y)
    if length is None:
        length = length_y
    if mean_y is None:
        mean_y = np.mean(y)
    # Wiener-Khinchin theorem
    transform_y = np.fft.rfft(y-mean_y, n=length + length_y)

    # see e.g. Allen&Tildesley regarding normalization
    norm = np.linspace(length_y, length_y-length+1, length)
    return np.fft.irfft(transform_y*np.conj(transform_y))[:length]/norm

def crosscorrelation(y, z, mean_y=None, mean_z=None, length=None):
    """Returns the crosscorrelation of two arrays.

    Make sure that the length of the arrays is an even number
    if speed is an issue."""
    length_data = len(y)
    if length is None:
        length = length_data
    if mean_y is None:
        mean_y = np.mean(y)
    if mean_z is None:
        mean_z = np.mean(z)
    # Wiener-Khinchin theorem
    transform_y = np.fft.rfft(y-mean_y, n=length+length_data)
    transform_z = np.fft.rfft(z-mean_z, n=length+length_data)
    return np.fft.irfft(transform_y*np.conj(transform_z))[:length]/length_data


def get_fast_fft_length(length):
    """Return the smallest power of 2 larger than 'length' for the fft."""
    return 1 << (length-1).bit_length()


class AnalyzeEnergies(object):
    """A class for analyzing the RUMD energy output files.

    The object is initialized by reading the metadata from energies0000.dat.gz.
    No energy data are read, andno statistics such as the mean and the variance.
    """

    def __init__(self, directory="TrajectoryFiles", energies_basefilename="energies", first_block=0, last_block=-1):
        import os

        self.t_dir = os.path.normpath(directory)
        self.metadata = dict()
        self.mean = dict()
        self.var = dict()
        self.covar = dict()
        self.energies = dict() # data arrays
        self.energies_basefilename = energies_basefilename
        
        self.read_energies_comment_line()
        self.metadata['first_block'] = first_block

        self.read_last_block()
        if last_block >= 0:
            if last_block < self.metadata['last_block']:
                self.metadata['last_block'] = last_block
            else:
                print('last_block specified  is greater than the last block actually present; using the latter')
            
    def read_energies_comment_line(self):
        """Read metadata from <energies base filename>0000.dat.gz comment line."""
        f = gzip.open(self.t_dir + '/%s0000.dat.gz' % (self.energies_basefilename), 'r')
        metadata = f.readline().decode().split()[1:]
        try:
            metadata = dict([s.split('=') for s in metadata])
            self.metadata['num_part'] = int(metadata['N'])
            self.metadata['interval'] = float(metadata['Dt'])
            self.metadata['column_keys'] = metadata['columns'].split(',')
        except ValueError:
            self.metadata['column_keys'] = metadata
            self.metadata['interval'] = 1
        print("found %u columns:" % len(self.metadata['column_keys']), end=' ')
        print(','.join(self.metadata['column_keys']))


    def read_last_block(self):
        """Read number of blocks from LastComplete_<energies base filename>.txt."""
        f = open(self.t_dir + '/LastComplete_%s.txt' % (self.energies_basefilename), 'r')
        self.metadata['last_block'] = int(f.readline().split(' ')[0])
        print("found %u blocks" % self.metadata['last_block'])


    def read_stats(self):
        """Read stats from rumd_stats output files."""
        f = open('%s_mean.dat % self.energies_basefilename', 'r')
        keys = f.readline().split()[1:]
        values = [float(s) for s in f.readline().split()]
        self.mean = dict(zip(keys, values))

        f = open('%s_var.dat' % self.energies_basefilename, 'r')
        keys = f.readline().split()[1:]
        values = [float(s) for s in f.readline().split()]
        self.var = dict(zip(keys, values))

        f = open('%s_covar.dat' %self.energies_basefilename, 'r')
        keys = f.readline().split()[1:]
        values = [float(s) for s in f.readline().split()]
        self.covar = dict(zip(keys, values))


    def compute_stats(self):
        """Do rumd_stats."""
        import rumd.Tools
        stats = rumd.Tools.rumd_stats()
        stats.ComputeStats()
        self.mean = stats.GetMeanVals()
        mean_sq = stats.GetMeanSqVals()
        self.covar = stats.GetCovarianceVals()
        for key in self.metadata['column_keys']:
            self.var[key] = mean_sq[key] - self.mean[key]**2


    def read_energies(self, column_names):
        """Read columns from energy files.

        This function reads one or more columns from the energy files and
        returns them as lists. A list of the names of the columns to be read
        should be given argument. The data are stored with the column
        identifiers as a dictionary in \"self.energies\".
        """
        column_numbers = []
        data = []
        for name in column_names:
            column_numbers.append(self.metadata['column_keys'].index(name))
            data.append([])
        print("Reading data from block %d to block %d" % (self.metadata['first_block'], self.metadata['last_block']))
        for block in range(self.metadata['first_block'], self.metadata['last_block'] + 1):
            f = gzip.open(self.t_dir+"/%s%0.4d.dat.gz"%(self.energies_basefilename, block), "r")
            f.readline() # Remove comment line
            for line in f:
                line = line.decode().split(' ')
                for data_column, energies_column in enumerate(column_numbers):
                    data[data_column].append(float(line[energies_column]))
        for i, name in enumerate(column_names):
            self.energies[name] = np.array(data[i])


    def correlation_function(self, key_a, key_b=None,
                             length=None, normalize=False):
        """Returns the auto- or crosscorrelation function.

        Calculate the autocorrelation function of one, or the cross-correlation
        of two columns in the energy files. Column labels should be given as
        arguments. Two arrays of equal length are returned, the first being the
        time, and the second being the correlation function.
        """
        # Check what has to be read from the energy files
        column_list = []
        if key_a not in self.energies:
            column_list.append(key_a)
        if key_b is not None and key_b not in self.energies:
            column_list.append(key_b)
        if len(column_list) > 0:
            self.read_energies(column_list)

        # make sure relevant statistics are present
        if key_a not in self.mean:
            self.mean[key_a] = np.mean(self.energies[key_a])
        if key_b is not None and key_b not in self.mean:
            self.mean[key_b] = np.mean(self.energies[key_b])

        # Choose the right padding length (smallest power of 2)
        length_data = len(self.energies[key_a])
        if length is None:
            length = length_data
        length_fast = get_fast_fft_length(length+length_data)
        length_pad = length_fast - length_data
        extra_pad = length_pad-length
        if extra_pad > 0:
            print("padded data with %u extra zeros for speed" % extra_pad)

        # Compute auto- or crosscorrelation, normalize if necessary
        if key_b is None:
            correlation = autocorrelation(self.energies[key_a],
                                          mean_y=self.mean[key_a],
                                          length=length_pad)[:length]
            if normalize:
                if key_a not in self.var:
                    self.var[key_a] = np.var(self.energies[key_a])
                correlation /= self.var[key_a]
                print("normalized autocorrelation function")
        else:
            correlation = crosscorrelation(self.energies[key_a],
                                           self.energies[key_b],
                                           mean_y=self.mean[key_a],
                                           mean_z=self.mean[key_b],
                                           length=length)
            if normalize:
                if key_a not in self.var or key_b not in self.var:
                    self.var[key_a] = np.var(self.energies[key_a])
                    self.var[key_b] = np.var(self.energies[key_b])
                correlation /= math.sqrt(self.var[key_a]*self.var[key_b])

        time = self.metadata['interval'] * np.arange(length)
        return time, correlation


    def response_function(self, key_a, key_b=None):
        """Return a linear response function.

        Given one ore two column names, the linear response function of those
        data sets are calculated. Two numpy arrays with the omega values
        and the response function are returned.
        """
        length_data = len(self.energies[key_a])
        if key_b is None:
            c = self.correlation_function(key_a)[1]
        else:
            c = self.correlation_function(key_a, key_b)[1]
        length_fft = get_fast_fft_length(2*length_data)
        c = np.gradient(c, self.metadata['interval'])
        c = np.fft.rfft(c, n=length_fft)[:length_data]
        c *= -self.metadata['num_part'] * self.metadata['interval']
        omega = np.fft.fftfreq(length_fft,
                               self.metadata['interval'])[:length_data]
        omega *= 2 * math.pi
        return omega, c
