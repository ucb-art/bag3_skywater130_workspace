# BAG3 SkyWater130 Template Workspace 

## Requirements 

This template workspace has only been tested on BWRC Linux servers thus far. A more comprehensive release of BAG3 with more nicely packaged dependencies is coming soon. In the meantime, please report any issues from missing dependencies.

The following C++ dependencies and versions were used. Note that more recent versions of these tools may be compatible but older versions may not be:
* CMake 3.17.0
* GCC 8.3.1 (Through devtoolset-8)
* Boost 1.72.0
* fmt 8.22

The following python dependencies are used. This workspace has been tested with Python 3.7.7:
* numpy 1.20.1
* scipy 1.6.2
* matplotlib 3.3.4
* hdf5 1.10.4


## Initial Setup 

After cloning, do the following:

1. ssh to a server/use a machine that supports RHEL7.
2. Start bash
3. Clone this repository and `cd` into it
4. Initialize all submodules by running `git submodule update --init --recursive`
5. Go through `.bashrc` and `.bashrc_bag`, and make sure all paths that start with `/path/to/` are updated to your a valid installation on your machine. Versions for tools are not strict, except for the `CMake` version. 
6. `source .bashrc`
7. Create a new cds.lib and set up symbolic links by running `./init_files.sh`
8. Compile pybag:
    - `cd BAG_framework/pybag`
    - `./run_test.sh`
    - `cd ../..`
9. Launch `virtuoso`. In the CIW, run `load('start_bag.il')`
    - This opens up the SKILL interface for BAG to import/export schematics and layouts
10. Run BAG commands from the bash shell

**Note**: Steps 3-5 and 7 only need to be done when creating a new BAG workspace.

**Note**: Step 8 needs to be done when creating a new BAG workspace or when pybag is updated.

For typical operation (i.e., with a BAG workspace that is already set up and no pybag updates), do the following:

1. Log into to any compute server that supports RHEL7
    - All machines on the LSF cluster should now support RHEL7
2. Start bash
3. `cd` into the workspace
4. `source .bashrc`
5. Launch `virtuoso`. In the CIW, run `load('start_bag.il')`
    - This opens up the SKILL interface for BAG to import/export schematics and layouts
6. Run BAG commands from the bash shell
## Caveats 

If you would like to use abstract generation, there are a few issues to be aware of. More details available of the issues surrounding abstract generation and additional setup instructions in [the tech plugin abstract setup README.](skywater130/abstract_setup/README.md)

## Licensing

This library is licensed under the Apache-2.0 license.  See [here](LICENSE) for full text of the
Apache license.
