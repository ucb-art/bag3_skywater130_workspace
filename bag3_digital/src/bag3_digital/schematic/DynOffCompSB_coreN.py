# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompSB_coreN(Module):
    """Module for library bag3_digital cell DynOffCompSB_coreN.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompSB_coreN.yaml')))

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
            w_dict='Dict of widths for different devices.',
            th_dict='Dict of intents for different devices.',
            dum_info='Dummy devices information.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dum_info=None,
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_dict: Dict[str, int],
               th_dict: Dict[str, str], dum_info: List[Tuple[Any]]) -> None:
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

        for _name, _str in zip(['XM1', 'XM2', 'XMc1', 'XMc2'], ['gm2n', 'gm2n', 'gm1', 'gm1']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        for _name, _str in zip(['XM3', 'XM4', 'XS3', 'XS4'], ['gm2p', 'gm2p', 'sw34', 'sw34']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        for _name in ['XS3_l', 'XS4_l', 'XS3_r', 'XS4_r']:
            self.design_transistor(_name, w_dict['sw34'], lch, seg_dict['sw34'] // 2,
                                   th_dict['sw34'], m='')

        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
