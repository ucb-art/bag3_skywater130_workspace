# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__sampler2_core(Module):
    """Module for library bag3_digital cell sampler2_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'sampler2_core.yaml'))

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
            seg_dict='Dict of segments for different devices.',
            lch='Channel length.',
            w='Width for different devices.',
            th='Intent for different devices.',
            dum_info='Dummy devices information.',
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w: int, th: str,
               dum_info: List[Tuple[Any]]) -> None:
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
        for _name, _str in zip(['XSIG0', 'XSIG1', 'XREF0', 'XREF1'], ['sig0', 'sig1', 'ref0',
                                                                      'ref1']):
            self.design_transistor(_name, w, lch, seg_dict[_str], th, m='')

        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
