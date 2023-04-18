# BAG3 SkyWater130 Template Workspace 

## Requirements 
**Note**: These steps only need to be run once to use BAG3. As of writing these instructions have only been tested on BWRC servers.

1. Install (on CentOS or Red Hat versions >=7) 
    * httpd24-curl
    * httpd24-libcurl
    * devtoolset-8 (compilers)
    * rh-git29 (better git)
    
3. Clone this repository and `cd` into it

4. Initialize all submodules by running `git submodule update --init --recursive`

5. Create and activate the conda environment from the provided `environment.yml`. Note the path to the solved environment.

6. The steps 4-12 involve installing other dependencies that have not been incorporated into the conda build
   Create a directory to install programs in (referred to as /path/to/programs).
   
7. Download and extract cmake 3.17.0:

	```
	 wget https://github.com/Kitware/CMake/releases/download/v3.17.0/cmake-3.17.0.tar.gz
	 tar -xvf cmake-3.17.0.tar.gz
	```
	 
	then build:

    ```
	cd cmake-3.17.0.tar.gz
    ./bootstrap --prefix=/path/to/conda/env/envname --parallel=4
    make -j4
    make install
    ```

8.  For magic\_enum:
    ```
    git clone https://github.com/Neargye/magic_enum.git
    cd magic_enum
    cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=FALSE \
        -DMAGIC_ENUM_OPT_BUILD_TESTS=FALSE -DCMAKE_INSTALL_PREFIX=/path/to/conda/env/envname
    cmake --build build
    cd build
    make install
    ```

9.  For yaml-cpp:
    ```
    git clone https://github.com/jbeder/yaml-cpp.git
    cd yaml-cpp
    cmake -B_build -H. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=/path/to/conda/env/envname
    cmake --build _build --target install -- -j 4
    ```

10.  For libfyaml:
    ```
    git clone https://github.com/pantoniou/libfyaml.git
    cd libfyaml
    ./bootstrap.sh
    ./configure --prefix=/path/to/conda/env/envname
    make -j12
    make install
    ```

11.  Download HDF5 1.10 (h5py-2.10 does not work with 1.12 yet)
     ```
     wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.6/src/hdf5-1.10.6.tar.gz
     tar -xvf hd5f-1.10.6.tar.gz
     ```
     , then install with:
	
     ```
     cd hd5f-1.10.6.tar.gz
     ./configure --prefix=/path/to/conda/env/envname
     make -j24
     make install
     ```


12.  Install Boost in steps 12-15. Download source, unzip.  In directory, run:

     ```
     wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz
     tar -xvf boost_1_72_0.tar.gz
     cd boost_1_72_0
     ./bootstrap.sh --prefix=/path/to/conda/env/envname
     ```

13.  Change the `using python` line to:

     ```
     using python : 3.7 : /path/to/conda/env/envname : /path/to/conda/env/envname/include/python3.7m ;
     ```
	
14.  Delete the line:
     ```
     path-constant ICU_PATH : /usr ;
     ```

15.  Run:

     ```
     ./b2 --build-dir=_build cxxflags=-fPIC -j8 -target=shared,static --with-filesystem --with-serialization --with-program_options install | tee install.log
     ```

     Remember to check install.log to see if there's any error messages (like python build error, etc.). We are not building with mpi
	
	
16.  In .bashrc_bag , set 
     ```
     export BAG_TOOLS_ROOT=/path/to/conda/env/envname
     ```
     and
     ```
     export BAG_TEMP_DIR=/scratch/path
     ```
	
17.  In .bashrc set
     ```
     export CMAKE_HOME=/path/to/programs/cmake-3.17.0 
     ```
     and change the Skywater PDK install directory to be your own
     
     ```
     export SW_PDK_ROOT=/tools/commercial/skywater
     ```

18. Test bag compilation by following steps the next section (Initial Setup). If you have issues upon compiling BAG, reinstall fmt>7.2 in conda, and spdlog in conda

## Initial Setup 

After cloning, do the following:

1. ssh to a server/use a machine that supports RHEL7.
2. Start bash
3. Clone this repository and `cd` into it (if you have not done so already).
4. Initialize all submodules by running `git submodule update --init --recursive` (if you have not done so already).
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
6. Run BAG commands from the bash shell. To test generation of an inverter cell, run  the following command inside the BAG workspace (folder this repo was cloned into): 
   ```
   ./gen_cell.sh data/bag3_digital/specs_blk/inv_chain/gen.yaml
   ```
   A corresponding library named `AAA_INV_CHAIN` will appear in your Virtuoso library manager. 

**Note**: A read-the-docs page is in progress for a more comprehensive introduction to BAG capabilities and instructions to make your own generators. 

## Licensing

This library is licensed under the Apache-2.0 license.  See [here](LICENSE) for full text of the
Apache license.
