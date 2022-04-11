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

import argparse
from pathlib import Path

from bag.io.file import read_yaml
from bag.util.misc import register_pdb_hook

from bag3_liberty.parse import read_liberty, write_liberty, LibObject, LibFun

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Remove generated_clock from libert files.')
    parser.add_argument('specs', help='YAML specs file name.')
    args = parser.parse_args()
    return args


def _set_max_val(obj: LibObject, trf_max: float, trf_detect: float, precision: int) -> bool:
    fmt_str = '{:%dg}' % precision
    changed = False
    for item in obj.objects:
        if isinstance(item, LibFun) and item.name == 'values':
            new_args = []
            for line in item.args:
                val_list = [float(v) for v in line[1:-1].split(', ')]
                for idx, v in enumerate(val_list):
                    if v > trf_detect:
                        changed = True
                        val_list[idx] = trf_max
                val_str_list = [fmt_str.format(v) for v in val_list]
                new_args.append(f'"{", ".join(val_str_list)}"')
            item.args = new_args
    return changed


def _transform_timing(obj: LibObject, trf_max: float, trf_detect: float, precision: int) -> bool:
    changed = False
    for item in obj.objects:
        if (isinstance(item, LibObject) and
                (item.dtype == 'rise_transition' or item.dtype == 'fall_transition')):
            changed |= _set_max_val(item, trf_max, trf_detect, precision)

    return changed


def _transform_pin(obj: LibObject, trf_max: float, trf_detect: float, precision: int) -> None:
    changed = False
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'timing':
            changed |= _transform_timing(item, trf_max, trf_detect, precision)
    if changed:
        print(f'Changed pin {obj.name} rise/fall max value to {trf_max}')


def _transform_cell(obj: LibObject, trf_max: float, trf_detect: float, precision: int) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject):
            if item.dtype == 'pin':
                _transform_pin(item, trf_max, trf_detect, precision)
            elif item.dtype == 'bus':
                for pin_item in item.objects:
                    if isinstance(pin_item, LibObject) and pin_item.dtype == 'pin':
                        _transform_pin(pin_item, trf_max, trf_detect, precision)


def _set_trf_max(obj: LibObject, trf_max: float, trf_detect: float, precision: int) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'cell':
            _transform_cell(item, trf_max, trf_detect, precision)


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    input_dir: str = specs['input_dir']
    output_dir: str = specs['output_dir']
    prefix: str = specs['prefix']
    trf_max: float = specs['trf_max']
    trf_detect: float = specs['trf_detect']
    precision: int = specs['precision']

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
            _set_trf_max(obj, trf_max, trf_detect, precision)
            write_liberty(out_file, obj)


if __name__ == '__main__':
    _args = parse_options()
    run_main(_args)
