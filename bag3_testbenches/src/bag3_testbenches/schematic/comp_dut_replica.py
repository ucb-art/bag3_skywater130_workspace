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

from typing import Dict, Any, Optional, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_testbenches__comp_dut_replica(Module):
    """Module for library bag3_testbenches cell comp_dut_replica.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'comp_dut_replica.yaml')))

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
            dut_cell='DUT cell',
            dut_params='DUT parameters',
            dut_conns='DUT connections dictionary',
            rep_conns='Replica connections dictionary',
            extra_pins='Extra pins to be added to wrapper',
            remove_pins_list='List of pins to be removed',
            dcfeed_array='Dictionary of dc_feed connections',
            dcblock_array='Dictionary of dc_block connections',
            noConn_array='Dictionary of noConn connections',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dut_conns=None,
            rep_conns=None,
            extra_pins=None,
            remove_pins_list=[],
            dcfeed_array=None,
            dcblock_array=None,
            noConn_array=None,
        )

    def design(self, dut_cell: str, dut_params: Param, dut_conns: Optional[Dict[str, str]],
               rep_conns: Optional[Dict[str, str]], extra_pins: Optional[Dict[str, list]],
               remove_pins_list: List[str], dcfeed_array: Optional[Dict[str, Any]],
               dcblock_array: Optional[Dict[str, Any]], noConn_array: Optional[Dict[str, Any]]
               ) -> None:
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
        if rep_conns is None:
            rep_conns = {}

        # setup DUT
        self.replace_instance_master('XDUT', 'bag3_digital', dut_cell, keep_connections=True)
        self.instances['XDUT'].design(**dut_params)
        for term_name, net_name in dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term_name, net_name)

        # setup replica
        self.replace_instance_master('XREP', 'bag3_digital', dut_cell, keep_connections=True)
        self.instances['XREP'].design(**dut_params)
        for term_name, net_name in rep_conns.items():
            self.reconnect_instance_terminal('XREP', term_name, net_name)

        # Pins
        for pin in remove_pins_list:
            self.remove_pin(pin)
        if extra_pins is not None:
            for pin_name in extra_pins['in']:
                self.add_pin(pin_name, TermType.input)
            for pin_name in extra_pins['out']:
                self.add_pin(pin_name, TermType.output)

        # DC feed
        if dcfeed_array is not None:
            for name, inst_term_list in dcfeed_array.items():
                self.array_instance(name, inst_term_list=inst_term_list)
        else:
            self.remove_instance('XDCf')

        # DC block
        if dcblock_array is not None:
            for name, inst_term_list in dcblock_array.items():
                self.array_instance(name, inst_term_list=inst_term_list)
        else:
            self.remove_instance('XDCb')

        # noConn
        if noConn_array is not None:
            for name, inst_term_list in noConn_array.items():
                self.array_instance(name, inst_term_list=inst_term_list)
        else:
            self.remove_instance('XNC')
