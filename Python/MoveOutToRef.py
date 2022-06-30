#!/usr/bin/python3
"""
Script to rename "out" timing files from performance Tests to corresponding "ref" files, ie replace the existing reference files by the newest output files.

"""

import os
import sys

suffix = ""

if len(sys.argv) > 1:
    suffix = sys.argv[1]

all_files = os.listdir(".")

refFiles = []
outFiles = []
for f in all_files:
    if not f.startswith("pe"):
        continue


    if suffix != "" and not f[:-4].endswith(suffix):
        continue

    if f.endswith(".out"):
        outFiles.append(f[:-4])
    #elif f.endswith(".ref"):
    #    refFiles.append(f[:-4])

outFiles.sort()

nFiles = len(outFiles)
for fdx in xrange(nFiles):
    basename = outFiles[fdx]

    mvCmd = "mv -f " + basename+".out " + basename+".ref"
    print(mvCmd)
    os.system(mvCmd)
