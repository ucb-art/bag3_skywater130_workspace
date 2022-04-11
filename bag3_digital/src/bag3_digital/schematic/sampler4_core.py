# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__sampler4_core(Module):
    """Module for library bag3_digital cell sampler4_core.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'sampler4_core.yaml'))

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
            ideal_switch='True to use ideal switch.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ideal_switch=False,
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w: int, th: str,
               dum_info: List[Tuple[Any]], ideal_switch: bool) -> None:
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
        if ideal_switch:
            self.replace_with_ideal_switch('XSIG')
            self.replace_with_ideal_switch('XREF')
        else:
            for _name, _str in zip(['XSIG', 'XREF'], ['sig', 'ref']):
                self.design_transistor(_name, w, lch, seg_dict[_str], th, m='')

        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')
