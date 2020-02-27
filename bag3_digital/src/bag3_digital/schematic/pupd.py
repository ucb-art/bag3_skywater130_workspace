# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__pupd(Module):
    """Module for library bag3_digital cell pupd.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'pupd.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            lch='channel length',
            w_p='pmos width.',
            w_n='nmos width.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            stack_p='stack of pmos',
            stack_n='stack of nmos',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            stack_p=1,
            stack_n=1,
        )

    def design(self, seg_p: int, seg_n: int, lch: int, w_p: int, w_n: int, th_p: str, th_n: str,
               stack_p: int, stack_n: int) -> None:
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
        self.design_transistor('XP', w_p, lch, seg_p, th_p, stack=stack_p)
        self.design_transistor('XN', w_n, lch, seg_n, th_n, stack=stack_n)
