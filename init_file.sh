#!/usr/bin/env bash

# BSD 3-Clause License

# Copyright (c) 2018, Regents of the University of California
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

if [ -z ${BAG_TEMP_DIR} ]
then
    echo "BAG_TEMP_DIR is unset. Source .bashrc first"
    exit 1
fi

# create cds.lib
echo "INCLUDE cds.lib.core" >> cds.lib

# set up symlinks
ln -s ${BAG_TEMP_DIR}/simulations/gen_outputs gen_outputs_scratch
ln -s ${BAG_TEMP_DIR}/calibre_run calibre_run

# setup cadence shared library linkage
if [ -z ${CDS_INST_DIR} ]
then
    echo "CDS_INST_DIR is unset"
    exit 1
fi

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
