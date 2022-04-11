# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynamicOffsetComp_Cap(Module):
    """Module for library bag3_digital cell DynamicOffsetComp_Cap.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'DynamicOffsetComp_Cap.yaml'))

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
            mode='1 for single ended; 2 for differential'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            mode=2,
        )

    def design(self, cap_in_params: Dict[str, Any], cap_cross_params: Dict[str, Any],
               doc_params: Dict[str, Any], mode: int) -> None:
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
