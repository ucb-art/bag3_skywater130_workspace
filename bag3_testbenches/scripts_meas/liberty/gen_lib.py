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

from typing import Dict, Any

import argparse

from bag.io import read_yaml


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate libert file from spec file.')
    parser.add_argument('lib_specs', help='Liberty Libert configuration specs file name.')
    parser.add_argument('cell_specs', help='Cell specs file name.')
    args = parser.parse_args()
    return args


def run_main(args: argparse.Namespace) -> None:
    lib_specs: Dict[str, Any] = read_yaml(args.lib_specs)
    cell_specs: Dict[str, Any] = read_yaml(args.cell_specs)

    cell_list = []
    lib_info = dict(lib_config=lib_specs, cells=cell_list)




if __name__ == '__main__':
    _args = parse_options()

    run_main(_args)
