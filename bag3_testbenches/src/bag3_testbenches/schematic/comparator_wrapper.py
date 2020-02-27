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

from typing import Dict, Any, Optional, List, Tuple, Union

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_testbenches__comparator_wrapper(Module):
    """Module for library bag3_testbenches cell comparator_wrapper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'comparator_wrapper.yaml')))

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
            dut_conns='Transistor DUT connection dictionary',
            dut_params='DUT parameters if it is a generator',
            extra_pins='Extra pins to be added to wrapper',
            remove_pins_list='List of pins to be removed',
            is_static='True if DUT is a static cellview',
            remove_cap_list='List of caps to be removed',
            array_cap_dict='Dict of caps to be arrayed',
            inst_par_val='instance, param, values to be updated',
            cap_conns='Dict of updated caps connections',
            invchain_params='Parameters for Inverter Chain',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dut_conns=None,
            extra_pins=None,
            is_static=False,
            remove_pins_list=[],
            remove_cap_list=[],
            inst_par_val=[],
            array_cap_dict=None,
            cap_conns=None,
        )

    def design(self, dut_lib: str, dut_cell: str, dut_conns: Optional[Dict[str, str]],
               extra_pins: Optional[Dict[str, list]], is_static: bool,
               dut_params: Optional[Param], remove_pins_list: List[str],
               remove_cap_list: List[str], inst_par_val: List[Tuple[str, str, Union[str, float]]],
               array_cap_dict: Optional[Dict[str, List[Tuple[str, List[Tuple[str, str]]]]]],
               cap_conns: Optional[Dict[str, Dict[str, str]]], invchain_params: Param) -> None:
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
        # design this testbench
        if dut_conns is None:
            dut_conns = {}

        # setup DUT
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=is_static,
                                     keep_connections=True)
        if not is_static:
            self.instances['XDUT'].design(**dut_params)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

        # Pins
        for pin in remove_pins_list:
            self.remove_pin(pin)
        if extra_pins is not None:
            for pin_name in extra_pins['in']:
                self.add_pin(pin_name, TermType.input)
            for pin_name in extra_pins['out']:
                self.add_pin(pin_name, TermType.output)

        # Inverter chain for clock signals
        self.instances['XClkInvChain'].design(**invchain_params)
        self.instances['XClkbInvChain'].design(**invchain_params)

        # Capacitors
        for cap_inst in remove_cap_list:
            self.remove_instance(cap_inst)

        if array_cap_dict is not None:
            for cap_name, inst_term_list in array_cap_dict.items():
                self.array_instance(cap_name, inst_term_list=inst_term_list)

        if cap_conns is not None:
            for cap_name, conns in cap_conns.items():
                for term_name, net_name in conns.items():
                    self.reconnect_instance_terminal(cap_name, term_name, net_name)

        for name, par, val in inst_par_val:
            self.instances[name].set_param(par, val)
