#!/usr/bin/python3
"""
Script to compare one timing files with another, where the new format is used
(one file holds data for all runs on a given card).
"""

import sys

if len(sys.argv) not in [3, 4]:
    print("Usage python3 <path>/CompareTimingsNew.py file1 file2 [unnormalized]")
    sys.exit(1)

use_unnormalized = "unnormalized" in sys.argv

filename1 = sys.argv[1]
filename2 = sys.argv[2]
file1 = open(filename1)
file2 = open(filename2)



next1 = file1.readline()
next2 = file2.readline()

min_time1 = None
min_time2 = None

while next1:
    assert next2

    while next1.startswith("#"):
        next1 = file1.readline()

    while next2.startswith("#"):
        next2 = file2.readline()

    items1 = next1.split()
    items2 = next2.split()

    # first three numbers are integers, use as index
    index1 = [int(item) for item in items1[:3]]
    index2 = [int(item) for item in items2[:3]]

    if not index1 == index2:
        print("Indices don't match:")
        print(index1)
        print(index2)
        sys.exit(1)

    # last number is normalized time
    time1, time1N = [float(item) for item in items1[-2:]]
    time2, time2N = [float(item) for item in  items2[-2:]]

    if min_time1 is None:
        min_time1 = time1
        min_time2 = time2
    else:
        if time1 < min_time1:
            min_index1 = index1[:]
            min_time1 = time1
            min_time1N = time1N
        if time2 < min_time2:
            min_index2 = index2[:]
            min_time2 = time2
            min_time2N = time2N


    if use_unnormalized:
        percentChange = 100.* ((time2 / time1) - 1.)
    else:
        percentChange = 100.* ((time2N / time1N) - 1.)

    print(index1, "%.1f %%" % percentChange)


    next1 = file1.readline()
    next2 = file2.readline()

print("Fastest in", filename1, "is", min_index1, "time:", min_time1N, "micro-sec/particle/step")
print("Fastest in", filename2, "is", min_index2, "time:", min_time2N, "micro-sec/particle/step")
