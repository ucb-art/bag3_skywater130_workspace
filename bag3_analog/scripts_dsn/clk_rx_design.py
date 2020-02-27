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
import argparse

from bag.io import read_yaml
from bag.core import BagProject

from pprint import pprint

from bag3_analog.design.clk_rx.clk_rx_des import ClkRXDesign


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
    parser = argparse.ArgumentParser(description='Clock Receiver design script.')
    parser.add_argument('specs', help='YAML specs file name.')
    parser.add_argument('-v', '--lvs', dest='run_lvs', action='store_true', default=False,
                        help='run LVS.')
    parser.add_argument('-mod', '--gen-model', dest='gen_mod', action='store_true', default=False,
                        help='generate behavioral model files.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    clk_rx_des = ClkRXDesign(specs)

    des_dict = clk_rx_des.design_clk_rx_closed_loop(prj)
    print('--- Final design ---')
    pprint(des_dict)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    run_main(_prj, _args)
