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

from bag3_testbenches.measurement.comparator.comp_des_LTI import ComparatorDesignLTI


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
    parser = argparse.ArgumentParser(description='Design comparator.')
    parser.add_argument('specs', help='Comparator design specs file name.')
    args = parser.parse_args()
    return args


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)

    comp_des = ComparatorDesignLTI(specs)
    comp_des.design_inv_split(vbias=0.5, cap_ratio=5.0, mode='single')


if __name__ == '__main__':
    _args = parse_options()

    run_main(_args)
