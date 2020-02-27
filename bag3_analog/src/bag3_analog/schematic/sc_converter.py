# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__sc_converter(Module):
    """Module for library bag3_analog cell sc_converter.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'sc_converter.yaml')))

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
            seg_dict='Dict of segments for different devices',
            lch='Channel length',
            w_dict='Dict of widths for different devices',
            th_dict='Dict of intents for different devices',
            dum_info='Dummy devices information',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dum_info=None,
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_dict: Dict[str, int],
               th_dict: Dict[str, str], dum_info: List[Tuple[Any]]) -> None:

        for _name, _str in zip(['XSW_n0', 'XSW_n1', 'XSW_n2', 'XSW_p0'],
                               ['nsw0', 'nsw1', 'nsw2', 'psw0']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
