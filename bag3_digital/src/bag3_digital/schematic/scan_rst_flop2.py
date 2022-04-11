# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__scan_rst_flop2(Module):
    """Module for library bag3_digital cell scan_rst_flop2.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'scan_rst_flop2.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:

        return dict(
            m_params='Master Parameters',
            s_params='Slave Parameters',
            inv_params='Inverter Params',
            dual_output='True to export out and outb',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            inv_params=None,
            dual_output=True,
        )

    def design(self, m_params: Param, s_params: Param, inv_params: Param, dual_output: bool
               ) -> None:
        self.instances['XM'].design(**m_params)
        self.instances['XS'].design(**s_params)
        self.instances['XClk'].design(**inv_params)
        self.instances['XScan'].design(**inv_params)

        if dual_output:
            self.remove_instance('XNC')
        else:
            self.remove_pin('outb')
            self.reconnect_instance_terminal('XNC', 'noConn', 'outb')
