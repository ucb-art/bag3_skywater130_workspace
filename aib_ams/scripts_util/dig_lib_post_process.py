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

from typing import Sequence, Set

import argparse
from pathlib import Path

from bag.io.file import read_yaml
from bag.util.misc import register_pdb_hook

from bag3_liberty.parse import read_liberty, write_liberty, LibObject

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Remove generated_clock from libert files.')
    parser.add_argument('specs', help='YAML specs file name.')
    args = parser.parse_args()
    return args


def _remove_gen_clk(obj: LibObject, white_set: Set[str]) -> None:
    for item in obj.objects:
        if isinstance(item, LibObject) and item.dtype == 'cell':
            new_objects = []
            changed = False
            for sub_item in item.objects:
                if isinstance(sub_item, LibObject) and sub_item.dtype == 'generated_clock':
                    cur_name = sub_item.name
                    if cur_name in white_set:
                        print(f'Keep generated clock: {cur_name}')
                        new_objects.append(sub_item)
                    else:
                        print(f'Remove generated clock: {cur_name}')
                        changed = True
                else:
                    new_objects.append(sub_item)
            if changed:
                item.objects = new_objects


def run_main(args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    input_dir: str = specs['input_dir']
    output_dir: str = specs['output_dir']
    prefix: str = specs['prefix']
    gen_clk_white_list: Sequence[str] = specs['gen_clk_white_list']

    white_set = set(gen_clk_white_list)
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
            _remove_gen_clk(obj, white_set)
            write_liberty(out_file, obj)


if __name__ == '__main__':
    _args = parse_options()
    run_main(_args)
