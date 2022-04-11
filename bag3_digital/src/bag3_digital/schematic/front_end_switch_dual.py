# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__front_end_switch_dual(Module):
    """Module for library bag3_digital cell front_end_switch_dual.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'front_end_switch_dual.yaml')))

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
            seg_dict='Dictionary of segments for different devices',
            lch='Transistor channel length, in resolution units',
            w_dict='Dictionary of transistor widths, in number of fins',
            th_dict='Dictionary of transistor threshold intent',
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_dict: Dict[str, int],
               th_dict: Dict[str, str]) -> None:
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

        for _name, _str in zip(['XSIG', 'XREF'], ['sig', 'ref']):
            self.design_transistor(_name + '_left', w_dict[_str], lch, seg_dict[_str],
                                   th_dict[_str])
            self.design_transistor(_name + '_right', w_dict[_str], lch, seg_dict[_str],
                                   th_dict[_str])

