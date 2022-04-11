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

from typing import Dict, Any, List, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_testbenches__comparator_tb(Module):
    """Module for library bag3_testbenches cell comparator_tb.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'comparator_tb.yaml')))

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
            dut_lib='Transistor DUT library name',
            dut_cell='Transistor DUT cell name',
            vbias_dict='Additional bias voltage dictionary',
            ibias_dict='Additional bias current dictionary',
            dut_conns='Transistor DUT connection dictionary',
            enable_pss='True to enable PSS simulation',
            enable_pac='True to enable PAC simulation',
            remove_inst_list='List of instances to be removed',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            vbias_dict=None,
            ibias_dict=None,
            dut_conns=None,
            enable_pss=False,
            enable_pac=False,
            remove_inst_list=[],
        )

    def design(self, dut_lib: str, dut_cell: str, vbias_dict: Optional[Dict[str, List[str]]],
               ibias_dict: Optional[Dict[str, List[str]]], remove_inst_list: List[str],
               dut_conns: Optional[Dict[str, str]], enable_pss: bool, enable_pac: bool) -> None:

        # design this testbench
        if vbias_dict is None:
            vbias_dict = {}
        if ibias_dict is None:
            ibias_dict = {}
        if dut_conns is None:
            dut_conns = {}

        # setup bias sources
        self.design_dc_bias_sources(vbias_dict, ibias_dict, 'VBIAS', 'IBIAS', define_vdd=False)

        # setup DUT
        self.replace_instance_master('XDUT_wrap', dut_lib, dut_cell, static=True,
                                     keep_connections=True)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT_wrap', term_name, net_name)

        # remove extra instances
        for inst in remove_inst_list:
            self.remove_instance(inst)

        # PSS settings
        if enable_pss:
            self.instances['VINPUT'].set_param('pwlperiod', 'pwl_per')
        if enable_pac:
            if enable_pss is False:
                raise ValueError('enable_pss should be True when enable_pac is True')
            self.instances['VINPUT'].set_param('pacm', 1)
