# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__diffamp_self_biased_buffer(Module):
    """Module for library bag3_analog cell diffamp_self_biased_buffer.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'diffamp_self_biased_buffer.yaml')))

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
            diffamp_params='Schematic parameters for self biased diffamp',
            buf_params='Schematic parameters for buffer',
            export_mid='True to export mid'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(export_mid=False)

    def design(self, diffamp_params: Param, buf_params: Param, export_mid: bool) -> None:
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
        self.instances['XDIFF'].design(**diffamp_params)
        self.instances['XBUF'].design(**buf_params)
        if not export_mid:
            self.remove_pin('v_mid')
