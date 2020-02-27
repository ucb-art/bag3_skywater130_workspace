# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompSB_CapN_split(Module):
    """Module for library bag3_digital cell DynOffCompSB_CapN_split.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompSB_CapN_split.yaml')))

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
            cap_in_params='Parameters for MOMCap connected to gates of pMOS',
            cap_p_params='Parameters for MOMCap connected to gates of pMOS',
            cap_n_params='Parameters for MOMCap connected to gates of nMOS',
            doc_params='Parameters for DOC comparator',
            mode='1 for single-ended; 2 for differential',
            nmos_switch='True to replace pmos switches with nmos switches',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            mode=2,
            nmos_switch=False,
        )

    def design(self, cap_p_params: Param, cap_n_params: Param, cap_in_params: Param,
               doc_params: Param, mode: int, nmos_switch: bool) -> None:
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
        # comparator design
        doc_params = doc_params.copy(append=dict(nmos_switch=nmos_switch))
        self.instances['XDOC'].design(**doc_params)
        if nmos_switch:
            for old_name, new_name in [('RSTB', 'RST'), ('RSTBD', 'RSTD')]:
                self.rename_pin(old_name, new_name)
                self.reconnect_instance_terminal('XDOC', new_name, new_name)

        # capacitors connected to gates of nMOS
        for cap in ['XCc1', 'XCc2']:
            self.instances[cap].design(**cap_n_params)

        # capacitors connected to gates of pMOS
        for cap in ['XCc3', 'XCc4']:
            self.instances[cap].design(**cap_p_params)

        # capacitors connected to input nMOS
        for cap in ['XCpreP', 'XCpreM']:
            self.instances[cap].design(**cap_in_params)

        if mode == 2:
            pass
        elif mode == 1:
            self.remove_pin('VIM')
            self.reconnect_instance_terminal('XCpreM', 'plus', 'VOM')
        else:
            raise ValueError(f'Unknown mode = {mode}. Use 1 for single ended; 2 for differential')
