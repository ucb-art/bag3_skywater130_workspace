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
from bag.simulation.base import setup_corner


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
    parser = argparse.ArgumentParser(description='Generate libert file from spec file.')
    parser.add_argument('lib_config', help='Library configuration yaml file name.')
    parser.add_argument('outdir', help='output directory name.')
    parser.add_argument('cells', nargs='+', help='Cell specification yaml file name.')
    parser.add_argument('-f', '--fake', dest='fake', action='store_true', default=False,
                        help='generate fake liberty file.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    lib_config = read_yaml(args.lib_config)
    outdir = args.outdir
    cell_list = [read_yaml(cell) for cell in args.cells]

    corner_yaml = read_yaml('data/aib_ams/lib_gen/ip_blocks/lib_corner_setup.yaml')

    base_name = os.path.basename(outdir)
    if base_name:
        lib_config_name = os.path.splitext(base_name)[0]
    else:
        lib_config_name = os.path.split(os.path.dirname(outdir))[-1]
    # libname_corner_voltage_othervoltage_temp

    corners_list = corner_yaml['corners_list']
    for corner in corners_list:
        corner_str = corner['name']
        voltage_str = str(corner['nom_voltage'])
        temp = str(corner['temp'])
        voltages = corner['voltages']
        vddio = str(voltages['VDDIO'])
        lib_name = lib_config_name + '_' + corner_str + '_' + voltage_str + '_' + vddio + '_' + temp

        outfile = outdir + '/' + lib_name
        lib_config['name'] = lib_name
        lib_config['sim_envs'][0]['temperature'] = int(temp)
        lib_config['sim_envs'][0]['voltage'] = float(voltage_str)
        lib_config['sim_envs'][0]['name'] = lib_name
        lib_config['voltages'].update(voltages)

        sim_env_str = setup_corner(corner_str, temp)
        for cell_params in cell_list:
            cell_params['sim_envs'] = [sim_env_str]

        generate_liberty(prj, lib_config, cell_list, outfile, fake=args.fake)


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