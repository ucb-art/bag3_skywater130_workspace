# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import argparse

from bag.io import read_yaml

from bag3_testbenches.liberty.io import generate_liberty
from bag.core import BagProject


def _info(etype, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(etype, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(etype, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


sys.excepthook = _info


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate liberty files needed for AIB AMS')
    parser.add_argument('outdir', help='output file name.')
    parser.add_argument('-f', '--fake', dest='fake', action='store_true', default=False,
                        help='generate fake liberty file.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    cell_base_dir = 'data/aib_ams/lib_gen/ip_blocks/'
    lib_config_file = cell_base_dir + 'lib_config.yaml'
    lib_config = read_yaml(lib_config_file)
    outdir = args.outdir

    cells = {
        'frontend' : 'aibcr3_frontend',
        'dll_dlyline64': 'aibcr3_dll_dlyline64',
        'dcc_dlyline64': 'aibcr3_dcc_dlyline64',
        'dll_phase_interp': 'aibcr3_dll_interpolator',
        'dlycell_dcc': 'aibcr3_dlycell_dcc',
        'dcc_phase_interp': 'aibcr3_dcc_interpolator',
        'dlycell_dll': 'aibcr3_dlycell_dcc_comb',
        'dcc_helper': 'aibcr3_dcc_helper',
        'dcc_phasedet': 'aibcr3_dcc_phasedet'
    }

    for cell, outfile in cells.items():
        cell_fixed = cell_base_dir + cell + '.yaml'
        cell_list = [read_yaml(cell_fixed)]
        outfile_fixed = outdir + outfile
        generate_liberty(prj, lib_config, cell_list, outfile_fixed, fake=args.fake)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if '_prj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['_prj']

    run_main(_prj, _args)

