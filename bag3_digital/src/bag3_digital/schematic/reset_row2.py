# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__reset_row2(Module):
    """Module for library bag3_digital cell reset_row2.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'reset_row2.yaml')))

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
            seg_dict='Dict of segments of all transistors',
            lch='Channel length in resolution units',
            w_p='pch width in resolution units',
            w_n='nch width in resolution units',
            th_p='pch threshold flavor',
            th_n='nch threshold flavor',
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_p: int, w_n: int, th_p: str,
               th_n: str) -> None:
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
        for _name, _str in zip(['XOUT_p', 'XKEEP_p'],
                               ['inv', 'keep']):
            self.design_transistor(_name, w_p, lch, seg_dict[_str], th_p, m='')

        for _name, _str in zip(['XOUT_n', 'XEN_n', 'XIN_minus', 'XIN_plus', 'XKEEP_n'],
                               ['inv', 'en_n', 'inp', 'inp', 'keep']):
            self.design_transistor(_name, w_n, lch, seg_dict[_str], th_n, m='')

        # array stack transistor
        num_seg = seg_dict['inner_p']
        suffix = f'<{num_seg - 1}:0>'
        for _name, _port in zip(['XRSTBD', 'XEN_p'], ['S', 'D']):
            new_name = _name + suffix
            if new_name != _name + '<1:0>':
                self.rename_instance(_name + '<1:0>', new_name)
            self.design_transistor(new_name, w_p, lch, 1, th_p, m='')
            self.reconnect_instance_terminal(new_name, _port, 'stack' + suffix)
