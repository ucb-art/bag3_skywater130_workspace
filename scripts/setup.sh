#!/bin/bash
set -e 

# Initial Workspace Setup 
# 
# * Link necessary OA binaries 
# * Initialize git submodules 
# * Create cds.lib 
# * 


if [[ -d pybag ]] || [[ -d cadence_libs ]] 
then
    echo "Bailing from setup.sh: one or more of the directories to be generated is already present!" 
    echo "If this is an already-working workspace, run 'source scripts/work.sh' instead. "
    exit 1 
fi

# Ensure sufficient gcc & related tools 
source scl_source enable devtoolset-8 rh-git29 httpd24

# Install submodules 
git submodule update --init --recursive

# Set up the run environment 
source .bashrc

# Ensure cds.lib is present 
echo 'INCLUDE $BAG_WORK_DIR/cds.lib.core' > cds.lib



if [ -z ${CDS_INST_DIR} ]
then
    echo "CDS_INST_DIR is unset"
    exit 1
fi


# setup symlink to compiled pybag
mkdir -p BAG_framework/pybag/_build/lib
cd BAG_framework/pybag/_build/lib
ln -s ${BAG_TOOLS_ROOT}/pybag .
cd ../../../../

# setup symlink for files
##ln -s skywater130/workspace_setup/bag_config.yaml bag_config.yaml
##ln -s skywater130/workspace_setup/bag_submodules.yaml bag_submodules.yaml
##ln -s skywater130/workspace_setup/.bashrc .bashrc
##ln -s skywater130/workspace_setup/.bashrc_bag .bashrc_bag
##ln -s skywater130/workspace_setup/.cshrc .cshrc
##ln -s skywater130/workspace_setup/.cshrc_bag .cshrc_bag
##ln -s skywater130/workspace_setup/.cdsenv .cdsenv
##ln -s skywater130/workspace_setup/.cdsinit .cdsinit
##ln -s skywater130/workspace_setup/cds.lib.core cds.lib.core
##ln -s skywater130/workspace_setup/display.drf display.drf
##ln -s skywater130/workspace_setup/.gitignore .gitignore
##ln -s skywater130/workspace_setup/leBindKeys.il leBindKeys.il
##ln -s skywater130/workspace_setup/pvtech.lib pvtech.lib
##ln -s BAG_framework/run_scripts/start_bag_ICADV12d3.il start_bag.il
##ln -s BAG_framework/run_scripts/virt_server.sh virt_server.sh
##ln -s BAG_framework/run_scripts/run_bag.sh run_bag.sh
##ln -s BAG_framework/run_scripts/start_bag.sh start_bag.sh

# setup cadence shared library linkage
mkdir cadence_libs

declare -a lib_arr=("libblosc.so"
                    "libblosc.so.1"
                    "libblosc.so.1.11.4"
                    "libcdsCommon_sh.so"
                    "libcdsenvutil.so"
                    "libcdsenvxml.so"
                    "libcla_sh.so"
                    "libcls_sh.so"
                    "libdataReg_sh.so"
                    "libddbase_sh.so"
                    "libdrlLog.so"
                    "libfastt_sh.so"
                    "libgcc_s.so"
                    "libgcc_s.so.1"
                    "liblz4.so"
                    "liblz4.so.1"
                    "liblz4.so.1.7.1"
                    "libnffr.so"
                    "libnmp_sh.so"
                    "libnsys.so"
                    "libpsf.so"
                    "libsrr_fsdb.so"
                    "libsrr.so"
                    "libstdc++.so"
                    "libstdc++.so.5"
                    "libstdc++.so.5.0.7"
                    "libstdc++.so.6"
                    "libstdc++.so.6.0.22"
                    "libvirtuos_sh.so"
                    "libz_sh.so"
                   )

for libname in "${lib_arr[@]}"; do
    fpath=${CDS_INST_DIR}/tools/lib/64bit/${libname}
    if [ ! -f "$fpath" ]; then
        echo "WARNING: Cannot find packaged Virtuoso shared library ${fpath}; symlink will be broken."
    fi
    ln -s ${fpath} cadence_libs/${libname}
done
