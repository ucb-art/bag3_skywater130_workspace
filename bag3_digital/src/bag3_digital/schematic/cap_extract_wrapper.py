# -*- coding: utf-8 -*-

from typing import Dict, Any, Sequence

import pkg_resources
from pathlib import Path

from pybag.enum import TermType

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__cap_extract_wrapper(Module):
    """Module for library bag3_digital cell cap_extract_wrapper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'cap_extract_wrapper.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            buf_params='Buffer parameters.',
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            in_pins='DUT input pins.',
            out_pins='DUT output pins.',
            io_pins='DUT inout pins.',
            extract_pin='cap extraction pin name.',
            buf_pwr='buffer pwr supply name.',
            buf_gnd='buffer ground name.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            buf_pwr='VDD',
            buf_gnd='VSS',
        )

    def design(self, buf_params: Param, dut_lib: str, dut_cell: str, in_pins: Sequence[str],
               out_pins: Sequence[str], io_pins: Sequence[str], extract_pin: str,
               buf_pwr: str, buf_gnd: str) -> None:
        self.instances['XBUF'].design(dual_output=False, **buf_params)
        self.instances['XREF'].design(dual_output=False, **buf_params)

        buf_conn_list = [('VDD', buf_pwr), ('VSS', buf_gnd)]
        self.reconnect_instance('XREF', buf_conn_list)
        self.reconnect_instance('XBUF', buf_conn_list)

        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)
        conn_list = []
        for pin in in_pins:
            if pin == extract_pin:
                conn_list.append((pin, 'cap_extract_match_'))
            else:
                self.add_pin(pin, TermType.input)
                conn_list.append((pin, pin))
        for pin in out_pins:
            conn_list.append((pin, pin))
            self.add_pin(pin, TermType.output)
        has_vdd = has_vss = False
        for pin in io_pins:
            conn_list.append((pin, pin))
            if pin == 'VDD':
                has_vdd = True
            elif pin == 'VSS':
                has_vss = True
            else:
                self.add_pin(pin, TermType.inout)
        if not has_vdd:
            self.remove_pin('VDD')
        if not has_vss:
            self.remove_pin('VSS')

        self.reconnect_instance('XDUT', conn_list)
