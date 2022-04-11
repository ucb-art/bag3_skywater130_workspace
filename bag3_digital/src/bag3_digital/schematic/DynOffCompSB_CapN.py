# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompSB_CapN(Module):
    """Module for library bag3_digital cell DynOffCompSB_CapN.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompSB_CapN.yaml')))

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
            ideal_switch='True to use ideal switches',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ideal_switch=False,
        )

    def design(self, cap_in_params: Dict[str, Any], cap_cross_params: Dict[str, Any],
               doc_params: Dict[str, Any], ideal_switch: bool) -> None:
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
        if ideal_switch:
            self.replace_instance_master('XDOC', 'bag3_digital',
                                         'DynOffCompSB_IdealSwitch_coreN',
                                         keep_connections=True)
        self.instances['XDOC'].design(**doc_params)

        self.instances['XCc1'].design(**cap_in_params)

        self.instances['XCc4'].design(**cap_cross_params)
