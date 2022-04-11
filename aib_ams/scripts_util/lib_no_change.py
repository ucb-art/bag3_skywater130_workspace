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

from typing import Sequence, List

import argparse
from pathlib import Path

from bag.io.file import read_yaml
from bag.util.misc import register_pdb_hook

from bag3_liberty.parse import read_liberty, write_liberty, LibObject, LibAttr, LibFun

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Remove generated_clock from libert files.')
    parser.add_argument('specs', help='YAML specs file name.')
    args = parser.parse_args()
    return args


def _add_offset(obj: LibObject, offset: float, precision: int) -> LibObject:
    obj_list = []
    fmt_str = '{:%dg}' % precision
    for item in obj.objects:
        if isinstance(item, LibFun) and item.name == 'values':
            new_args = []
            for line in item.args:
                val_list = [fmt_str.format(float(v) + offset) for v in line[1:-1].split(', ')]
                new_args.append(f'"{", ".join(val_list)}"')
            obj_list.append(LibFun('values', new_args))
        else:
            obj_list.append(item)

    return LibObject(obj.name, obj.dtype, obj_list)


def _fill_timing_info(table: Sequence[Sequence[List[LibObject]]], obj: LibObject,
                      guard_high: bool, clk_period: float, precision: int) -> str:
    ttype = ''
    related = ''
    data = [None, None]
    for item in obj.objects:
        if isinstance(item, LibAttr):
            if item.name == 'timing_type':
                ttype = item.val.strip()
            if item.name == 'related_pin':
                related = item.val.strip()
        elif isinstance(item, LibObject):
            if item.dtype == 'fall_constraint':
                data[0] = item
            elif item.dtype == 'rise_constraint':
                data[1] = item
            else:
                raise ValueError(f'Unknown data dtype: {item.dtype}')

    if not ttype:
        raise ValueError('Cannot find timing type')
    if not related:
        raise ValueError('Cannot find related pin')

    if ttype == 'setup_falling':
        if guard_high:
            raise ValueError('Not supported yet')
        else:
            if data[0] is not None:
                table[0][guard_high].append(data[0])
            if data[1] is not None:
                table[1][guard_high].append(data[1])
    elif ttype == 'setup_rising':
        if guard_high:
            if data[0] is not None:
                table[0][guard_high].append(data[0])
            if data[1] is not None:
                table[1][guard_high].append(data[1])
        else:
            if data[0] is not None:
                table[0][guard_high].append(_add_offset(data[0], -clk_period / 2, precision))
            if data[1] is not None:
                table[1][guard_high].append(_add_offset(data[1], -clk_period / 2, precision))
    elif ttype == 'hold_falling':
        if guard_high:
            if data[0] is not None:
                table[1][guard_high].append(data[0])
            if data[1] is not None:
                table[0][guard_high].append(data[1])
        else:
            raise ValueError('Not supported yet')
    elif ttype == 'hold_rising':
        if guard_high:
            raise ValueError('Not supported yet')
        else:
            if data[0] is not None:
                table[1][guard_high].append(data[0])
            if data[1] is not None:
                table[0][guard_high].append(data[1])
    else:
        raise ValueError(f'Unknown timing type: {ttype}')

    return related


def _change_pin(name: str, detect_list: Sequence[str]) -> bool:
    for detect in detect_list:
        if detect in name:
            return True
    return False


def _transform_pin(obj: LibObject, pin_detect_list: Sequence[str],
                   guard_high, clk_period, precision: int) -> None:
    if not _change_pin(obj.name, pin_detect_list):
        return

    print(f'transforming pin: {obj.name}')
    timing_table = [[[], []], [[], []]]
    new_obj_list = []
    related = ''
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'timing':
            new_related = _fill_timing_info(timing_table, item, guard_high, clk_period, precision)
            if not related:
                related = new_related
            elif related != new_related:
                raise ValueError('Conflict in related pin')
        else:
            new_obj_list.append(item)

    for data_lev in [0, 1]:
        data_str = 'low' if data_lev == 0 else 'high'
        for rel_lev in [0, 1]:
            rel_str = 'low' if rel_lev == 0 else 'high'
            key = f'nochange_{data_str}_{rel_str}'
            time_list = timing_table[data_lev][rel_lev]
            if time_list:
                time_list.insert(0, LibAttr('timing_type', key))
                time_list.insert(0, LibAttr('related_pin', related))
                new_obj_list.append(LibObject('', 'timing', time_list))

    obj.objects = new_obj_list


def _transform_cell(obj: LibObject, pin_detect_list: Sequence[str],
                    guard_high: bool, clk_period: float, precision: int) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject):
            if item.dtype == 'pin':
                _transform_pin(item, pin_detect_list, guard_high, clk_period, precision)
            elif item.dtype == 'bus':
                for pin_item in item.objects:
                    if isinstance(pin_item, LibObject) and pin_item.dtype == 'pin':
                        _transform_pin(pin_item, pin_detect_list, guard_high, clk_period, precision)


def _to_no_change(obj: LibObject, pin_detect_list: Sequence[str],
                  guard_high: bool, clk_period: float, precision: int) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'cell':
            _transform_cell(item, pin_detect_list, guard_high, clk_period, precision)


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    input_dir: str = specs['input_dir']
    output_dir: str = specs['output_dir']
    prefix: str = specs['prefix']
    pin_detect_list: Sequence[str] = specs['pin_detect_list']
    guard_high: bool = specs['guard_high']
    clk_period: float = specs['clk_period']
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
            _to_no_change(obj, pin_detect_list, guard_high, clk_period, precision)
            write_liberty(out_file, obj)


if __name__ == '__main__':
    _args = parse_options()
    run_main(_args)
