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

from typing import Dict, Any, Optional, Union, List

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param
from bag.math import float_to_si_string


# noinspection PyPep8Naming
class bag3_testbenches__digital_timing_tb(Module):
    """Module for library bag3_testbenches cell digital_timing_tb.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'digital_timing_tb.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            dut_lib='dut_lib',
            dut_cell='dut_cell',
            dut_conns='pin connection',
            cout_conns='If None (the default) cout will be deleted, if given it should be a '
                       'mapping of terminals to tuple of nets or a single net',
            clk_params='clk parameters',
            vdd_dict='Dictionary describing name and value of vdd supplies',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            cout_conns=None,
            vdd_dict=None,
        )

    def design(self,
               dut_lib: str,
               dut_cell: str,
               dut_conns: Dict[str, str],
               cout_conns: Optional[Dict[str, Union[str, List[str]]]],
               clk_params: Dict[str, Any] = None,
               vdd_dict: Dict[str, Any] = None) -> None:
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        self.replace_instance_master('XDUT',
                                     lib_name=dut_lib,
                                     cell_name=dut_cell,
                                     static=True)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

        if vdd_dict:
            vdd_name_list = list(vdd_dict.keys())
            term_list = [dict(PLUS=x[0], MINUS=x[1]) for x in vdd_dict.values()]
            self.array_instance('VVDD', vdd_name_list, term_list)
            for name, params in vdd_dict.items():
                v = params[-1]
                if isinstance(v, int) or isinstance(v, float):
                    v = float_to_si_string(v)
                self.instances[name].set_param('vdc', v)

        if cout_conns:
            term_net_iter = []
            cout_name = 'Xcout'
            first_net_names = next(iter(cout_conns.values()))
            if isinstance(first_net_names, list):
                if len(first_net_names) > 1:
                    cout_name = f'Xcout<{len(first_net_names) - 1}:0>'
                    for term, nets in dut_conns.items():
                        term_net_iter.append((term, ','.join(nets)))
                else:
                    raise ValueError('If cout_conns is provided as a list the minimum len should '
                                     'be more than 1')
                self.rename_instance('Xcout', cout_name)
            if not term_net_iter:
                term_net_iter = cout_conns.items()
            self.reconnect_instance(cout_name, term_net_iter)
        else:
            self.delete_instance('Xcout')

        if clk_params:
            for key, val in clk_params.items():
                self.instances['Vin'].set_param(key, val)
