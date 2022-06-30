rumd_image   {#rumd_image}
===========================

Tool generating an image of a RUMD configuration using [POV-ray](http://www.povray.org).


Usage
--------
The `-h` option will generate the following:

```

  Convert RUMD configuration to a image using POV-ray [www.povray.org].

Usage: rumd_image [-i start.xyz.gz] [-o start]

  -h,         Print this help message.
  -f INT,     Restart frame index.
  -i FILE,    Input configuration.
                The default is start.xyz.gz.
  -p FILE,    Povray input header file.
                A default file is generated if file does not exist.
                The default is myPovray.ini.
  -o STR,     Output file name, i.e.
                will generate STR.pov and STR.png.
                STR=INPUT_NAME: name of the input configuration.
  -a NUM,     Camera angle position (radians).
                The default is -pi/8.
  -y NUM,     Camera height
                The default is 0.66*Ly
  -r NUM,     Camera radial distance
                The default is 2.0*Lx
  -L NUM,NUM,NUM,
              Camera look_at position
                The default is 0,-0.15*Ly,0
  -W INT,     Width of output image. Height is set to 3/4 of width.
                The default is 800 (height of 600).
  -d,         Disable running povray (but do generate *.pov file).

Usage example:
  rumd_image -d -i start.xyz.gz -o start
  povray +P +W1600 +H1200 +A +HImyPovray.ini +Istart.pov

Caveat: Default myPovray.ini file can only handle up to 16 particle types. 

Dependency: POV-ray [www.povray.org].

```


Usage examples
----------------------
Below is an image of a configuration of 864 Lennard-Jones particles.

![Image made with rumd_image](LJ864.png)


### How to generate movie of a configuration

A movie of a configuration can be generated using [ffmpeg](http://www.ffmpeg.org/)
together with `rumd_image` and [POV-ray](http://www.povray.org/). Generate a POV-ray
scene where the position of the camera is given by the value `clock` that will vary 
from 0-1 during the movie.

1. Generate an example configuration (`start.xyz.gz`)

    rumd_init_conf

2. Generate a POV-ray scene where the angular position of the camera
   is set using the `clock` value that will range from 0-1.

    rumd_image -d -a 2*pi*clock -o scene

3. Generate 100 images for the movie using [POV-ray](http://www.povray.org/). 
   The option `+KFF100` dictates that 100 images should be made,
   `+W400 +H300` set the pixel size, and `+KC` that the movie can be looped.

    povray -D +W400 +H300 +HImyPovray.ini +Iscene.pov +KFF100 +KC +Oframe.png

4. Use [ffmpeg](https://www.ffmpeg.org/) to generate a movie from images

    ffmpeg -i frame%03d.png movie.mp4

5. Clean up by deleting images

    rm frame???.png

6. View movie, quit by pressing `q` (the `-loop 0` option makes the movie loop indefinitely)

    ffplay -loop 0 movie.mp4

Hint: You can edit `myPovray.ini` and `scene.pov` between step 2 and 3 to modify the scene.

Implementation
-----------------------
This tool is a bash script that read a RUMD *.xyz.gz file
and writes a file for [POV-ray](http://www.povray.org/).


See also
--------------------
\ref rumd_init_conf

\ref rumd_visualize

