# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__clk_rx_wrapper(Module):
    """Module for library bag3_analog cell clk_rx_wrapper.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'clk_rx_wrapper.yaml')))

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
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            export_mid='True to export output of diffamp before buffer',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(export_mid=True)

    def design(self, export_mid: bool, dut_lib: str, dut_cell: str) -> None:
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
        # replace DUT statically
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True, keep_connections=True)

        if not export_mid:
            self.remove_pin('v_mid')
