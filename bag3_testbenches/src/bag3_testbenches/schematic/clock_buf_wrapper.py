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

from typing import Dict, Any, Optional, List

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_testbenches__clock_buf_wrapper(Module):
    """Module for library bag3_testbenches cell clock_buf_wrapper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'clock_buf_wrapper.yaml')))

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
            dut_lib='DUT library name',
            dut_cell='DUT cell name',
            dut_conns='DUT connection dictionary',
            dut_params='Parameters for DUT',
            invchain_params='Parameters for InvChain',
            extra_pins='Extra pins to be added to wrapper',
            remove_pins_list='List of pins to be removed',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            extra_pins=None,
            remove_pins_list=[]
        )

    def design(self, dut_lib: str, dut_cell: str, dut_conns: Optional[Dict[str, str]],
               dut_params: Param, invchain_params: Param, extra_pins: Optional[Dict[str, list]],
               remove_pins_list: List[str]) -> None:
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
        # setup DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, keep_connections=True)
        self.instances['XDUT'].design(**dut_params)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

        # design inverter chains
        self.instances['XClkInvChain'].design(**invchain_params)
        self.instances['XClkbInvChain'].design(**invchain_params)

        # extra pins
        if extra_pins is not None:
            for pin_name in extra_pins['in']:
                self.add_pin(pin_name, TermType.input)
            for pin_name in extra_pins['out']:
                self.add_pin(pin_name, TermType.output)
        for pin in remove_pins_list:
            self.remove_pin(pin)
