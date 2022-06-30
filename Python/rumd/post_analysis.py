#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" Python tools to investigate a RUMD simulation.

Usage example:
    import rumd.post_analysis as pa

    df = pa.get_energies_as_DataFrame()
    print(f"Mean potential energy per particle: {df['pe'].mean()}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(df['t'], df['pe'], 'bo')
    plt.xlabel(r'Time, $t$ [$\sigma\sqrt{m/\varepsilon}$]')
    plt.ylabel(r'Potential energy per particle, $u$ [$\varepsilon$]')
    plt.show()
 
"""


def documentation_in_jupyter(func):
    ''' Pretty print of documentation in a Jupyter Notebook '''
    import docutils.core
    import docutils.writers.html5_polyglot
    from IPython.display import HTML

    writer = docutils.writers.html5_polyglot.Writer()
    str_data = func.__doc__
    return HTML(docutils.core.publish_string(str_data, writer=writer).decode('UTF-8'))


def get_index_last_completed(
        directory='./TrajectoryFiles/',
        filename='LastComplete_energies.txt'):
    """ Return index of last completed frame """
    from os.path import join
    with open(join(directory, filename), 'rt') as f:
        index_last_complete = int(f.readline().split()[0])
    return index_last_complete


def get_energies_column_names(directory='./TrajectoryFiles/'):
    ''' Return column names of the file
    with energetic data (energies0000.dat.gz).
    Return None if column names cannot be found. '''
    from os.path import join
    import gzip
    filename = join(directory, 'energies0000.dat.gz')
    with gzip.open(filename, 'rt') as f:
        header = f.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            if variable_name == 'columns':
                values = section.split('=')[1]
                column_names = values.split(',')
                return column_names
    return None


def get_energies_time_interval(directory='./TrajectoryFiles/'):
    ''' Return column names of the file with energetic data.
    Return None if column names cannot be found. '''
    from os.path import join
    import gzip
    filename = join(directory, 'energies0000.dat.gz')
    with gzip.open(filename, 'rt') as f:
        header = f.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            if variable_name == 'Dt':
                value = float(section.split('=')[1])
                return value
    return None


def get_energies(energy_variable='pe', directory='./TrajectoryFiles/',
                 return_column_names=False, return_Dt=False):
    """ Returns a list of energies of the energies{i:04d}.dat.gz files  """
    from os.path import join
    import gzip
    enr_list = []  # List of energies in column named energy_variable
    index_last_complete = get_index_last_completed(directory)
    # Read data in all the energy files
    for i in range(index_last_complete+1):
        filename = join(directory, f'energies{i:04d}.dat.gz')
        with gzip.open(filename, 'rt') as f:
            # Read the header of the file to get the column of interest
            header = f.readline().split()[1:]
            for section in header:
                variable_name = section.split('=')[0]
                values = section.split('=')[1]
                if variable_name == 'columns':
                    column_names = values.split(',')
                    if return_column_names:
                        return column_names
                if variable_name == 'Dt':
                    if return_Dt:
                        return float(section.split('=')[1])
            column_of_interest = -1
            for index, name in enumerate(column_names):
                if(name == energy_variable):
                    column_of_interest = index
            if(column_of_interest == -1):
                print(f'Columns in energy file: {column_names}')
                print(
                    f'Warning: column named {energy_variable} was not found.')
            # Read data in column of interest and append to list
            for line in f.read().splitlines():
                enr_list.append(float(line.split()[column_of_interest]))
    return enr_list


def get_energies_as_DataFrame(directory='./TrajectoryFiles/', verbose=False):
    """ Return energetic data as Pandas DataFrame """
    from os.path import join
    from pandas import read_csv, Series, concat
    from numpy import arange
    if verbose:
        print('Read energetic data as Pandas DataFrame.')
    column_names = get_energies_column_names(directory)
    index_last_complete = get_index_last_completed(directory)
    list_of_dataframes = []
    for i in range(index_last_complete+1):
        if verbose:
            print_to_frame_counter(i, 0)
        filename = join(directory, f'energies{i:04d}.dat.gz')
        df = read_csv(filename, sep=' ', skiprows=1,
                      header=None, names=column_names, index_col=False)
        list_of_dataframes.append(df)
    df = concat(list_of_dataframes, ignore_index=True)
    Dt = get_energies_time_interval(directory)
    times = Series(name='t', data=arange(0, len(df))*Dt)
    return concat([times, df], axis='columns')


def time_correlation(x, y=None):
    r''' Compute the time correlation function 
    The time correlation function :math:`C(t)` is computed using
    the Wiener–Khinchin theorem with the FFT algorithm and zero padding.

     .. math::

         C(t) = \langle \Delta x(\tau) \Delta y(\tau + t) \rangle_\tau

         \Delta x(t) = x(t) - \langle x \rangle

         \Delta y(t) = x(y) - \langle y \rangle

    '''
    from numpy import conj, mean, real
    from numpy.fft import fft, ifft
    if y is None:
        y = x
    n = x.size
    fx = fft(x-mean(x), n=2*n)
    fy = fft(y-mean(y), n=2*n)
    fxy = conj(fx)*fy/n
    xy = ifft(fxy)[:n]
    return real(xy)


def frequency_dependent_response(x, y=None, dt=1.0, prefactor=1.0):
    r''' Frequency dependent responce
    The frequency dependent responce :math:`\mu(\omega)` is estimate from time-series
    assuming the fluctuation-dissipation theorem.

     .. math::

         \mu(\omega) = A\int_0^\infty \dot C(t)\exp(-i \omega t) dt

         \dot C(t) = \frac{d}{dt} \langle \Delta x(\tau) \Delta y(\tau + t) \rangle_\tau

         \Delta x(t) = x(t) - \langle x \rangle

         \Delta y(t) = y(t) - \langle y \rangle

    dt is the sample time of the time-series.
    Notice that the user must provide a prefactor
    to get the correct scaling responce function.
    The prefactor (:math:`A`) is -1/kT**2
    for the frequency dependent heat capacity.
    If y is None then it is set to be the same as the x time-series.
    The implimentation uses the FFT algorithm
    with zero-padding for Laplace transform,
    thus the calculation scales as $N\ln(N)$.

    Returns::

        omega, mu

     '''
    from numpy import arange, pi
    n = x.size
    k = arange(0, n)  # k = 0, 1, ... , n-1
    omega = 2*pi*k/dt/n

    from numpy.fft import fft
    from numpy import gradient
    if y is None:
        y = x
    C = gradient(time_correlation(x, y))
    mu = prefactor*fft(C, n=2*n)[:n]
    return omega, mu


def run_avg(x, n=128):
    '''  Running average of n data points. '''
    from numpy import mean, zeros
    N = int(len(x)/n)
    out = zeros(N)
    for i in range(N):
        start = i*n
        stop = (i+1)*n
        out[i] = mean(x[start:stop])
    return out


def run_avg_log(x, points_per_decade=24, base=None):
    ''' Logarithmic averaging.  '''
    from numpy import mean, array
    if not base:
        base = 10**(1/points_per_decade)
    if base < 1:
        print('Warning: Base should be larger than 1')
    floor, ceil, next_ceil = 0, 1, 1.0
    out = []
    while ceil < x.size:
        out.append(mean(x[floor:ceil]))
        # Set limits for next average range
        floor = ceil
        while int(next_ceil) == ceil:
            next_ceil = next_ceil*base
        ceil = int(next_ceil)
    return array(out)


def get_number_of_particles(directory='./TrajectoryFiles/', frame=0):
    """ Return the number of particles """
    from os.path import join
    import gzip
    filename = join(directory, f'restart{frame:04d}.xyz.gz')
    with gzip.open(filename, 'rt') as f:
        number_of_particles = int(f.readline())
    return int(number_of_particles)


def get_box_size(directory='./TrajectoryFiles/', frame=0):
    """ The X, Y and Z box lengths of a restart file.

    Returns:
        X, Y, Z

    """
    from os.path import join
    import gzip
    filename = join(directory, f'restart{frame:04d}.xyz.gz')
    with gzip.open(filename, 'rt') as f:
        _ = f.readline()  # Skip line with number of particles
        header = f.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            values = section.split('=')[1]
            if variable_name == 'sim_box':
                column = values.split(',')
                X = float(column[1])
                Y = float(column[2])
                Z = float(column[3])
                return X, Y, Z
    return None


def get_box_volume(directory='./TrajectoryFiles/', frame=0):
    """ Return the volume of the simulation box of a restart file. """
    X, Y, Z = get_box_size(directory=directory, frame=frame)
    return X*Y*Z


def get_number_density(directory='./TrajectoryFiles/', frame=0):
    """ Return the number number density of a restart file. """
    from os.path import join
    import gzip
    filename = join(directory, f'restart{frame:04d}.xyz.gz')
    with gzip.open(filename, 'rt') as f:
        number_of_particles = int(f.readline())
        header = f.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            values = section.split('=')[1]
            if variable_name == 'sim_box':
                column = values.split(',')
                X = float(column[1])
                Y = float(column[2])
                Z = float(column[3])
                return number_of_particles/X*Y*Z
    return None


def get_restart_positions(directory='./TrajectoryFiles/', frame_index=0):
    """ Returns lists of position in the restart{i:04d}.xyz.gz files  """
    from os.path import join
    import gzip
    ptype = []  # Particle types
    x, y, z = [], [], []  # list with position
    X, Y, Z = None, None, None  # list of box vectors
    number_of_particles = None
    filename = join(directory, f'restart{frame_index:04d}.xyz.gz')
    with gzip.open(filename, 'rt') as f:
        number_of_particles = int(f.readline())
        header = f.readline().split()[1:]
        for section in header:
            variable_name = section.split('=')[0]
            values = section.split('=')[1]
            if variable_name == 'columns':
                column_names = values.split(',')
                for i, var in enumerate(column_names):
                    if(var == 'type'):
                        iptype = i
                    if(var == 'x'):
                        ix = i
                    if(var == 'y'):
                        iy = i
                    if(var == 'z'):
                        iz = i
            if variable_name == 'sim_box':
                column = values.split(',')
                X = float(column[1])
                Y = float(column[2])
                Z = float(column[3])
        for i in range(number_of_particles):
            column = f.readline().split(' ')
            ptype.append(int(column[iptype]))
            x.append(float(column[ix]))
            y.append(float(column[iy]))
            z.append(float(column[iz]))
    return ptype, (x, y, z), (X, Y, Z)


def get_neighbor_list(positions, boundary_box, r_cut=1.6):
    ''' Return a neighbur list. This is done using a cell list, so the
    scaling in N '''
    from math import floor
    xyz, XYZ = positions, boundary_box

    number_of_particles = len(xyz[0])
    Nxyz = [floor(XYZ[0]/r_cut), floor(XYZ[1]/r_cut), floor(XYZ[2]/r_cut)]
    # print("Cell list size:", Nxyz)

    def get_index(idx):
        ''' Return index in cell_list index from x,y and z coordinates (idx)
        Used to build neighboir list '''
        for dim in range(3):
            idx[dim] = idx[dim]-Nxyz[dim]*floor(idx[dim]/Nxyz[dim])
        return idx[2]*Nxyz[1]*Nxyz[0]+idx[1]*Nxyz[0]+idx[0]

    # Build cell list
    cell_list = [[] for i in range(Nxyz[0]*Nxyz[1]*Nxyz[2])]
    cell_index_of_n = [-1]*number_of_particles*3
    for n in range(number_of_particles):
        ixyz = [0, 0, 0]  # Cell list index in x, y, and z direction
        for dim in range(3):
            x = xyz[dim][n]/XYZ[dim]  # Reduce coordinte
            x = x-floor(x)             # Coordinate between 0 and 1
            ixyz[dim] = floor(x*Nxyz[dim])
            cell_index_of_n[n*3+dim] = ixyz[dim]
        cell_list[get_index(ixyz)].append(n)

    def append_pair_to_neighbour_list(n, m):
        ''' Add pair to neighbout list '''
        r2 = 0.0  # pair distance squared
        for dim in range(3):
            dx = xyz[dim][n] - xyz[dim][m]
            dx = dx - XYZ[dim]*round(dx/XYZ[dim])
            r2 += dx*dx
        if(r2 < r_cut**2 and n < m):  # Found neighbour
            neighbour_list[n].append(m)
            neighbour_list[m].append(n)

    # Build neighbour list
    neighbour_list = [[] for __ in range(number_of_particles)]
    if(min(Nxyz) > 2):  # Use cell list to build neighbour list
        for n in range(number_of_particles):
            ixyz = [0, 0, 0]
            for dix in range(-1, 2):  # Loop neighbour cells
                ixyz[0] = cell_index_of_n[n*3+0]+dix  # Index of neighbour cell
                for diy in range(-1, 2):
                    ixyz[1] = cell_index_of_n[n*3+1]+diy
                    for diz in range(-1, 2):
                        ixyz[2] = cell_index_of_n[n*3+2]+diz
                        for m in cell_list[get_index(ixyz)]:
                            append_pair_to_neighbour_list(n, m)
    else:  # Fall back on simple N^2 neighbour list build method
        for n in range(number_of_particles-1):  # loop pairs
            for m in range(n+1, number_of_particles):
                append_pair_to_neighbour_list(n, m)

    # Sort neighbour list (for esthetic reasons)
    for n in range(number_of_particles):
        neighbour_list[n].sort()

    return neighbour_list


def get_r2(npos, mpos, pb):
    ''' Return the square distance using the minimum image convension '''
    r2 = 0.0
    for dim in range(3):
        dx = npos[dim]-mpos[dim]
        dx = dx - pb[dim]*round(dx/pb[dim])
        r2 += dx*dx
    return r2


def get_radial_distribution(r_min=0.8, r_max=1.6, r_step=0.01,
                            directory='./TrajectoryFiles/',
                            frames='all', verbose=True):
    ''' Computes radial distribution function of restart files.
    Returns (r, rdf) where r and rdf are numpy arrays'''
    from numpy import zeros_like, sqrt, arange, pi
    r = arange(r_min, r_max, r_step)
    rdf = zeros_like(r)

    if frames == 'all':
        last_index = get_index_last_completed(directory,
                                              'LastComplete_restart.txt')
        frames = range(last_index)
    if verbose:
        print('Compute radial distribution')
    particle_counter = 0
    mean_density = 0

    for frame_index in frames:
        if verbose:
            print_to_frame_counter(frame_index, frames[0])
        ptype, xyz, XYZ = get_restart_positions(directory, frame_index)
        mean_density += len(xyz[0])/(XYZ[0]*XYZ[1]*XYZ[2])
        neighbor_list = get_neighbor_list(xyz, XYZ, r_max)
        for n, neighbors in enumerate(neighbor_list):
            particle_counter += 1
            for m in neighbors:
                npos = (xyz[0][n], xyz[1][n], xyz[2][n])
                mpos = (xyz[0][m], xyz[1][m], xyz[2][m])
                r2 = get_r2(npos, mpos, XYZ)  # pair distance squared
                r_bin = round((sqrt(r2) - r_min)/r_step)
                if(r_bin >= 0 and
                   r_bin < len(rdf) and
                   n < m):  # Found neighbour
                    rdf[r_bin] += 1
    mean_density /= len(frames)
    rdf = rdf/(2*pi*r*r*r_step)/particle_counter/mean_density
    return r, rdf


def get_clusters(directory='./TrajectoryFiles/',
                 frame_index=0,
                 r_cut=1.6,
                 minmass=1,
                 outputfiles=['clusters.xyz.gz',
                              'clustersGC.xyz.gz',
                              'clusters.csv']):
    ''' Find clusters of particles using a cut-off.
    Return a list of cluster list of cluster masses.
    Writes files to disks with particle informations '''
    from math import sqrt
    import gzip

    outputfile = outputfiles[0]
    outputfileGC = outputfiles[1]
    outputfileCSV = outputfiles[2]

    ptype, xyz, XYZ = get_restart_positions(directory, frame_index)
    number_of_particles = len(xyz[0])
    neighbour_list = get_neighbor_list(xyz, XYZ, r_cut)

    # Find clusters (using a stack)
    particle_stack = list(range(number_of_particles))
    current_cluster = -1
    cluster_assignment = [-1]*number_of_particles
    particles_in_cluster = []
    particles_to_check = 0
    know_particles_in_current_cluster = []
    while(len(particle_stack) > 0):
        if(particles_to_check == 0):  # This is a new cluster
            current_cluster += 1
            particles_in_cluster.append([])
            particles_to_check = 1
            know_particles_in_current_cluster = []
        # Pop the particle at the top of the stack and find
        #   new particles in cluster connecected to this
        current_particle = particle_stack.pop()
        particles_to_check -= 1
        cluster_assignment[current_particle] = current_cluster
        particles_in_cluster[-1].append(current_particle)
        # Put connecting particles (belonging to the current cluster),
        # on the top of the stack (will be checked later)
        for n in neighbour_list[current_particle]:
            for i, m in enumerate(particle_stack):
                if n == m and m not in know_particles_in_current_cluster:
                    particle_stack.append(particle_stack.pop(i))
                    particles_to_check += 1
                    know_particles_in_current_cluster.append(m)

    # Write xyz-file with cluster assignments
    print(f'Write particle assigments of clusters to {outputfile}')
    with gzip.open(outputfile, 'wt') as f:
        f.write(f'{number_of_particles}\n')
        f.write(
            f'Lattice="{XYZ[0]} 0.0 0.0 0.0 {XYZ[1]} 0.0 0.0 0.0 {XYZ[2]}" ')
        f.write(
            f'sim_box=RectangularSimulationBox,{XYZ[0]},{XYZ[1]},{XYZ[2]} ')
        f.write('columns=type,x,y,z,cluster,mass_of_my_cluster \n')
        for n in range(number_of_particles):
            t = ptype[n]
            x = xyz[0][n]
            y = xyz[1][n]
            z = xyz[2][n]
            cluster = cluster_assignment[n]
            mass_of_my_cluster = len(particles_in_cluster[cluster])
            f.write(f'{t} {x} {y} {z} {cluster} {mass_of_my_cluster}\n')

    # Compute how many clusters are larger than min mass
    number_of_large_clusters = 0
    for c in range(len(particles_in_cluster)):
        ns = particles_in_cluster[c]
        if len(ns) >= minmass:
            number_of_large_clusters += 1

    # Find and write a file with Geometric Centers (GC) of clusters
    print(f'Write Geometric Centers (GC) of clusters to {outputfileGC}')
    print(f'Clusters smaller than minmass = {minmass} are not included.')
    geometric_centers_of_clusters = [0]*len(particles_in_cluster)*3
    sqr_secound_moment_of_clusters = [
        0]*len(particles_in_cluster)  # sqrt(sum(dx)/N)
    with gzip.open(outputfileGC, 'wt') as f:
        f.write(f'{number_of_large_clusters}\n')
        f.write(
            f'Lattice="{XYZ[0]} 0.0 0.0 0.0 {XYZ[1]} 0.0 0.0 0.0 {XYZ[2]}" ')
        f.write(
            f'sim_box=RectangularSimulationBox,{XYZ[0]},{XYZ[1]},{XYZ[2]} ')
        f.write('columns=0,x,y,z,mass,sqr_secound_moment \n')
        for c in range(len(particles_in_cluster)):
            ns = particles_in_cluster[c]
            if len(ns) >= minmass:
                gc = [xyz[0][ns[0]],
                      xyz[1][ns[0]],
                      xyz[2][ns[0]]]  # Geometric center (use first particle)
                dgc = [1, 1, 1]  # Displacement of Geometric center
                th = 1e-6  # Threshold value for convergence of displacement
                sqr_secound_moment = 0.0  # std deviation of position
                while(dgc[0] > th and dgc[1] > th and dgc[2] > th):
                    dgc = [0.0, 0.0, 0.0]
                    sqr_secound_moment = 0.0
                    for n in particles_in_cluster[c]:
                        for dim in range(3):
                            dx = xyz[dim][n]-gc[dim]
                            dx = dx - XYZ[dim]*round(dx/XYZ[dim])
                            sqr_secound_moment += dx*dx
                            dgc[dim] += dx
                    sqr_secound_moment = sqrt(
                        sqr_secound_moment/len(particles_in_cluster[c]))
                    for dim in range(3):
                        gc[dim] += dgc[dim]/len(particles_in_cluster[c])
                        geometric_centers_of_clusters[3*c+dim] = gc[dim]
                        sqr_secound_moment_of_clusters[c] = sqr_secound_moment
                mass = len(particles_in_cluster[c])
                ssm = sqr_secound_moment
                f.write(f'{0} {gc[0]} {gc[1]} {gc[2]} {mass} {ssm}\n')

    # Write pandas DataFrane and CSV file with cluster information
    print(f'Write information about all clusters to {outputfileCSV}')
    from pandas import DataFrame
    df = DataFrame(columns=['id', 'mass'])
    with open(outputfileCSV, 'wt') as f:
        f.write('id,mass,sqr_secound_moment,x,y,z\n')
        for c in range(len(particles_in_cluster)):
            mass = len(particles_in_cluster[c])
            ssm = sqr_secound_moment_of_clusters[c]
            x = geometric_centers_of_clusters[3*c+0]
            y = geometric_centers_of_clusters[3*c+1]
            z = geometric_centers_of_clusters[3*c+2]
            f.write('%d,%d,%6.6f,%6.6f,%6.6f,%6.6f\n' %
                    (c, mass, ssm, x, y, z))
            df = df.append({
                'id': int(c),
                'mass': int(mass),
                'Squareroot of second moment': float(ssm),
                'x': x,
                'y': y,
                'z': z
            }, ignore_index=True)

    # Write CSV file with cluster information
    from numpy import average, std
    print("\n..:: Statistics on all clusters ::..")
    mlist = []  # List of masses
    for c in range(len(particles_in_cluster)):
        mlist.append(len(particles_in_cluster[c]))
    print("Number of clusters:", len(mlist))
    if len(mlist) > 0:
        print("Mass of largest cluster (all):", max(mlist))
        print("Mass of smallest cluster (all):", min(mlist))
        print("Average cluster mass (all):", average(mlist))
        print("Std deviation of cluster masses (all):", std(mlist))

    print(f"\n..:: Statistics on large clusters, m>={minmass} ::..")
    mlist = []  # List of masses
    for c in range(len(particles_in_cluster)):
        if len(particles_in_cluster[c]) >= minmass:
            mlist.append(len(particles_in_cluster[c]))
    print("Number of large clusters:", len(mlist))
    if len(mlist) > 0:
        print("Mass of largest cluster (large):", max(mlist))
        print("Mass of smallest cluster (large):", min(mlist))
        print("Average cluster mass (large):", average(mlist))
        print("Std deviation of cluster masses (large):", std(mlist))

    print(f"\n..:: Statistics on small clusters, m<{minmass} ::..")

    mlist = []  # List of masses
    for c in range(len(particles_in_cluster)):
        if len(particles_in_cluster[c]) < minmass:
            mlist.append(len(particles_in_cluster[c]))
    print("Number of small clusters:", len(mlist))
    if len(mlist) > 0:
        print("Mass of largest cluster (small):", max(mlist))
        print("Mass of smallest cluster (small):", min(mlist))
        print("Average cluster mass (small):", average(mlist))
        print("Std deviation of cluster masses (small):", std(mlist))

    return df


def scattering_function_of_one_frame(xs, X, nxs=range(1, 65)):
    """ Return the one dimentional scattering function """
    from numpy import cos, sin, pi
    k, S = [], []
    for nx in nxs:
        real = 0
        imag = 0
        for x in xs:
            real += cos(-2.0*pi*nx*x/X)
            imag += sin(-2.0*pi*nx*x/X)
        S.append((real * real + imag * imag) / len(xs))
        k.append(2.0*pi*nx/X)
    return k, S


def scattering_function(directory='./TrajectoryFiles/',
                        nxs=range(1, 65),
                        dimention=0,
                        first_frame=0,
                        last_frame=None):
    """ Return scattering function for wave-vectors along the x-direction """
    from numpy import zeros
    if last_frame is None:
        last_frame = get_index_last_completed(directory)
    S = zeros(len(nxs))
    for frame in range(first_frame, last_frame):
        xyz, XYZ = get_restart_positions(directory, frame=frame)
        x = xyz[dimention]
        X = XYZ[dimention]
        k, this_S = scattering_function_of_one_frame(x, X, nxs)
        S = S + this_S
        print_to_frame_counter(frame, first_frame)
    S = S/float(last_frame-first_frame)
    return k, S


def linear_regression(x, y):
    """Return the slope, intersection and the Person correlation coefficient
    of the input (x,y) data """
    from numpy import cov, mean, sqrt
    cov_matrix = cov(x, y)  # Covariance matrix
    slope = cov_matrix[0, 1]/cov_matrix[0, 0]
    intersection = mean(y) - slope*mean(x)
    R = cov_matrix[0, 1]/sqrt(cov_matrix[0, 0]*cov_matrix[1, 1])
    return slope, intersection, R


def print_to_frame_counter(frame=0, first_frame=0):
    """ Write pretty frame count to user """
    import datetime
    f = frame-first_frame
    if(f % 50 == 0):
        print(f'{frame:04d} |', end='')
    elif(f % 10 == 0):
        print('|', end='')
    elif(f % 5 == 0):
        print(':', end='')
    else:
        print('.', end='')
    if(f % 50 == 49):
        print(f'| {datetime.now().ctime()}', end='\n')


def main():
    ''' Main function when the file is runned as a script '''

    df = get_energies_as_DataFrame()
    print(f"Mean potential energy per particle: {df['pe'].mean()}")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(df['t'], df['pe'], 'bo')
    plt.xlabel(r'Time, $t$ [$\sigma\sqrt{m/\varepsilon}$]')
    plt.ylabel(r'Potential energy per particle, $u$ [$\varepsilon$]')
    plt.show()


if(__name__ == '__main__'):
    main()
