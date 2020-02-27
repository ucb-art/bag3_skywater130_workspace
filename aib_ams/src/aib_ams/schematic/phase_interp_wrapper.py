# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
from pybag.enum import TermType


class aib_ams__phase_interp_wrapper(Module):
    """Module for library aib_ams cell phase_interp_wrapper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'phase_interp_wrapper.yaml')))

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
            dc_params='Delay Cell params',
            dut_lib='DUT library name.',
            dut_cell='DUT cell name',
            nbits='Number of bits in the phase interpolator',
            b_in='have b_in',
            outb='have the outb exported'
        )

    def design(self, dc_params: Param, dut_lib: str, dut_cell: str, nbits: int, b_in: bool,
               outb: bool) -> None:
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
        sp = f'sp<{nbits-1}:0>'
        sn = f'sn<{nbits-1}:0>'
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True, keep_connections=False)
        self.rename_pin('sp', sp)
        self.rename_pin('sn', sn)
        for pin in [sp, sn, 'VDD', 'VSS', 'intout']:
            self.reconnect_instance_terminal('XDUT', pin, pin)
        self.reconnect_instance_terminal('XDUT', 'a_in', 'intin')
        if b_in:
            self.add_pin('b_in', TermType.output)
            self.reconnect_instance_terminal('XDUT', 'b_in', 'b_in')
        if outb:
            self.add_pin('outb', TermType.output)
            self.reconnect_instance_terminal('XDUT', 'outb', 'outb')
        self.add_pin('intin', TermType.output)
        self.instances['XDC'].design(**dc_params)
