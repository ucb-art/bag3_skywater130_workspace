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

from typing import Sequence

import argparse
from pathlib import Path

from bag.io.file import read_yaml
from bag.util.misc import register_pdb_hook

from bag3_liberty.parse import read_liberty, write_liberty, LibObject, LibAttr

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Remove generated_clock from libert files.')
    parser.add_argument('specs', help='YAML specs file name.')
    args = parser.parse_args()
    return args


def _change_pin(name: str, pin_list: Sequence[str]) -> bool:
    return name in pin_list


def _transform_pin(obj: LibObject, pin_list: Sequence[str]) -> None:
    if _change_pin(obj.name, pin_list):
        has_clock_attr = False
        for item in obj.objects:
            if isinstance(item, LibAttr) and item.name == 'clock':
                item.val = 'true'
                has_clock_attr = True
                break

        if not has_clock_attr:
            obj.objects.insert(0, LibAttr('clock', 'true'))


def _transform_cell(obj: LibObject, pin_list: Sequence[str]) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject):
            if item.dtype == 'pin':
                _transform_pin(item, pin_list)
            elif item.dtype == 'bus':
                for pin_item in item.objects:
                    if isinstance(pin_item, LibObject) and pin_item.dtype == 'pin':
                        _transform_pin(pin_item, pin_list)


def _add_is_clk(obj: LibObject, pin_list: Sequence[str]) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'cell':
            _transform_cell(item, pin_list)


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    input_dir: str = specs['input_dir']
    output_dir: str = specs['output_dir']
    prefix: str = specs['prefix']
    pin_list: Sequence[str] = specs['pin_list']

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f'reading lib files from {input_path}, writing to {output_path}')
    for file_path in input_path.glob(f'{prefix}*.lib'):
        print(f'transforming {file_path.name}')
        out_file = output_path / file_path.name
        try:
            obj = read_liberty(file_path)
        except ValueError as ex:
            print(f'Error parsing {file_path.name}:\n{str(ex)}')
            obj = None
        if obj is not None:
            _add_is_clk(obj, pin_list)
            write_liberty(out_file, obj)


if __name__ == '__main__':
    _args = parse_options()
    run_main(_args)
