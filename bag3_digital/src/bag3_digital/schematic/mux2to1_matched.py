# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__mux2to1_matched(Module):
    """Module for library bag3_digital cell mux2to1_matched.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'mux2to1_matched.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            inv_params='Output inverter parameters',
            tri_params='Tristate inverter parameters'
        )

    def design(self, inv_params: Param, tri_params: Param) -> None:
        self.instances['XPASS<1:0>'].design(**tri_params)
        self.instances['XBUF'].design(**inv_params)
        self.instances['XSUM'].design(nin=2)
