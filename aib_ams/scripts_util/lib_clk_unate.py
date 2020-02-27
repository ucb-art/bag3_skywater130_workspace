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

from typing import Mapping, Union

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


def _transform_pin(obj: LibObject, pin_map: Mapping[str, Union[str, Mapping[str, str]]]) -> None:
    time_sense = pin_map.get(obj.name, None)
    if time_sense is not None:
        for item in obj.objects:
            if isinstance(item, LibObject) and item.dtype == 'timing':
                new_obj_list = []
                rel_pin = ''
                sense_idx = 0
                for cur_idx, entries in enumerate(item.objects):
                    if isinstance(entries, LibAttr):
                        if entries.name != 'timing_sense':
                            if entries.name == 'related_pin':
                                rel_pin = (entries.val[1:-1]
                                           if entries.val[0] == '"' else entries.val)
                            new_obj_list.append(entries)
                        else:
                            sense_idx = cur_idx
                    else:
                        new_obj_list.append(entries)

                if not rel_pin:
                    raise ValueError('Cannot find related pin')

                if isinstance(time_sense, str):
                    sense = time_sense
                else:
                    sense = time_sense[rel_pin]
                print(f'Changing pin {obj.name} (related = {rel_pin}) timing sense to {sense}')

                new_obj_list.insert(sense_idx, LibAttr('timing_sense', sense))
                item.objects = new_obj_list


def _transform_cell(obj: LibObject, pin_map: Mapping[str, Union[str, Mapping[str, str]]]) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject):
            if item.dtype == 'pin':
                _transform_pin(item, pin_map)
            elif item.dtype == 'bus':
                for pin_item in item.objects:
                    if isinstance(pin_item, LibObject) and pin_item.dtype == 'pin':
                        _transform_pin(pin_item, pin_map)


def _change_clk_unate(obj: LibObject, pin_map: Mapping[str, Union[str, Mapping[str, str]]]) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'cell':
            _transform_cell(item, pin_map)


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    root_dir: str = specs['root_dir']
    cells: Mapping[str, Mapping[str, Union[str, Mapping[str, str]]]] = specs['cells']

    root_path = Path(root_dir)
    for cell_name, pin_map in cells.items():
        dir_path = root_path / cell_name

        print(f'reading lib files from {dir_path}')
        for file_path in dir_path.glob(f'{cell_name}_*.lib'):
            print(f'transforming {file_path.name}')
            out_file = dir_path / file_path.name
            try:
                obj = read_liberty(file_path)
            except ValueError as ex:
                print(f'Error parsing {file_path.name}:\n{str(ex)}')
                obj = None
            if obj is not None:
                _change_clk_unate(obj, pin_map)
                write_liberty(out_file, obj)


if __name__ == '__main__':
    _args = parse_options()
    run_main(_args)
