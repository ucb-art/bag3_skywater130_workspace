# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__sc_converter_array(Module):
    """Module for library bag3_analog cell sc_converter_array.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sc_converter_array.yaml')))

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
            ncol='Number of columns in array',
            nrow='Number of rows in array',
            unit_sch_params='Parameter of unit cell',
        )

    def design(self, ncol: int, nrow: int, unit_sch_params: Dict[str, Any]) -> None:

        total_inst_num = ncol * nrow
        inst_name = f'XCELL<{total_inst_num-1}:0>'
        self.instances['XCELL'].design(**unit_sch_params)
        self.array_instance('XCELL', inst_name_list=[inst_name])
