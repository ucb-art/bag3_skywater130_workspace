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

from typing import Union, Type, Dict, Any, TextIO, cast

import argparse

from pybag.core import BBox, get_wire_iterator

from bag.io.file import open_file, read_yaml
from bag.core import BagProject
from bag.util.importlib import import_class
from bag.util.misc import register_pdb_hook
from bag.layout.template import TemplateBase

register_pdb_hook()


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulate cell from spec file.')
    parser.add_argument('specs', help='YAML specs file name.')
    parser.add_argument('layer_list', help='list of layer numbers, separated by space.')
    parser.add_argument('out_file', help='output file name.')
    args = parser.parse_args()
    return args


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)
    impl_lib: str = specs['impl_lib']
    lay_str: Union[str, Type[TemplateBase]] = specs.get('lay_class', '')
    params: Dict[str, Any] = specs['params']

    layer_list = [int(val) for val in args.layer_list.split()]

    lay_cls = cast(Type[TemplateBase], import_class(lay_str))

    lay_db = prj.make_template_db(impl_lib)

    lay_master: TemplateBase = lay_db.new_template(lay_cls, params=params)
    res = lay_master.grid.resolution
    with open_file(args.out_file, 'w') as f:
        for name in lay_master.port_names_iter():
            port = lay_master.get_port(name)
            for lay_id in layer_list:
                pin_list = port.get_pins(lay_id)
                for pin in pin_list:
                    if isinstance(pin, BBox):
                        write_line(f, res, pin.xl, pin.yl, pin.xh, pin.yh, name)
                    else:
                        for lay, purp, bbox in get_wire_iterator(lay_master.grid,
                                                                 lay_master.tr_colors,
                                                                 pin.track_id, pin.lower,
                                                                 pin.upper):
                            write_line(f, res, bbox.xl, bbox.yl, bbox.xh, bbox.yh, name)


def write_line(f: TextIO, res: float, xl: int, yl: int, xh: int, yh: int, name: str) -> None:
    fmt = '{:.4f}'
    val_list = [fmt.format(xl * res), fmt.format(yl * res),
                fmt.format(xh * res), fmt.format(yh * res), name]
    f.write(','.join(val_list))
    f.write('\n')


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
