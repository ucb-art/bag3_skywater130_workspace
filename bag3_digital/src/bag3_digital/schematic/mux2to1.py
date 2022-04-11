# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__mux2to1(Module):
    """Module for library bag3_digital cell mux2to1.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'mux2to1.yaml')))

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
            sel_inv='Inverter params for select',
            out_inv='Output inverter params',
            tristate='Tristate inverter params'
        )

    def design(self, sel_inv: Param, out_inv: Param, tristate: Dict[str, Any]) -> None:
        self.instances['Xtr<1:0>'].design(**tristate)
        self.instances['Xinvout'].design(dual_output=False, **out_inv)
        self.instances['Xinvin'].design(dual_output=False, **sel_inv)
        self.instances['XCM'].design(nin=2)

