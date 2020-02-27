# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__bag3_digitalDB_test(Module):
    """Module for library bag3_digital cell bag3_digitalDB_test.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'bag3_digitalDB_test.yaml')))

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
            seg_list='',
            lch='',
            w_p='',
            w_n='',
            th_n='',
            th_p='',
            stack_n='',
            stack_p='',
        )

    def design(self, seg_list, lch, w_p, w_n, th_n, th_p, stack_n, stack_p) -> None:
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

        for i in range(3):
            inst_name = f'Xinv{i+1}'
            params = dict(
                seg=seg_list[i],
                lch=lch,
                w_p=w_p,
                w_n=w_n,
                th_n=th_n,
                th_p=th_p,
                stack_p=stack_p,
                stack_N=stack_n
            )
            self.instances[inst_name].design(**params)
