# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompVb_CapN(Module):
    """Module for library bag3_digital cell DynOffCompVb_CapN.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompVb_CapN.yaml')))

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
            cap_in_params='Parameters for input MOMCap',
            cap_cross_params='Parameters for cross MOMCap',
            doc_params='Parameters for DOC comparator',
            mode='1 for single ended; 2 for differential',
            nmos_switch='True to replace pmos switch with nmos switch',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            mode=2,
            nmos_switch=False,
        )

    def design(self, cap_in_params: Param, cap_cross_params: Param, doc_params: Param, mode: int,
               nmos_switch: bool) -> None:
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
        doc_params = doc_params.copy(append=dict(nmos_switch=nmos_switch))
        self.instances['XDOC'].design(**doc_params)

        self.instances['XCc1'].design(**cap_in_params)
        self.instances['XCc2'].design(**cap_in_params)

        self.instances['XCc3'].design(**cap_cross_params)
        self.instances['XCc4'].design(**cap_cross_params)

        if mode == 2:
            pass
        elif mode == 1:
            self.reconnect_instance_terminal('XCc2', 'minus', 'VOM')
            self.remove_pin('VpreM')
        else:
            raise ValueError(f'Unknown mode={mode}. Use 1 for single-ended, 2 for differential.')

        if nmos_switch:
            for old_pin, new_pin in [('RSTB', 'RST'), ('RSTBD', 'RSTD')]:
                self.reconnect_instance_terminal('XDOC', new_pin, new_pin)
                self.rename_pin(old_pin, new_pin)
