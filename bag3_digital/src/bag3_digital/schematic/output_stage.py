# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__output_stage(Module):
    """Module for library bag3_digital cell output_stage.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'output_stage.yaml')))

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
            latch_params='Parameters for SR latch',
            rstrow_params='Parameters for ResetRow',
            invnor2_params='Parameters for Inv->NOR2',
            nand2_params='Parameters for NAND2',
        )

    def design(self, latch_params: Dict[str, Any], rstrow_params: Dict[str, Any],
               invnor2_params: Dict[str, Any], nand2_params: Dict[str, Any]) -> None:
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
        self.instances['XNAND_left'].design(**nand2_params)
        self.instances['XNAND_right'].design(**nand2_params)

        self.instances['XInvNOR2'].design(**invnor2_params)

        self.instances['XRstRow'].design(**rstrow_params)

        self.instances['XSR'].design(**latch_params)
