# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompSB_coreN2(Module):
    """Module for library bag3_digital cell DynOffCompSB_coreN2.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompSB_coreN2.yaml')))

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
            sym_clk='True to have symmetrically delayed clock on left and right',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dum_info=None,
            sym_clk=False,
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_dict: Dict[str, int],
               th_dict: Dict[str, str], dum_info: List[Tuple[Any]], sym_clk: bool) -> None:
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

        # nch devices
        for _name, _str in zip(['XM1', 'XM2', 'XMc1', 'XMc2', 'XPOW'],
                               ['gm2n', 'gm2n', 'gm1', 'gm1', 'pow']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        # pch devices + pch switches
        for _name, _str in zip(['XM3', 'XM4', 'XS1', 'XS2', 'XS3', 'XS4'],
                               ['gm2p', 'gm2p', 'sw12', 'sw12', 'sw34', 'sw34']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        # charge injection cancellation devices
        for _name in ['XS3_l', 'XS4_l', 'XS3_r', 'XS4_r']:
            self.design_transistor(_name, w_dict['sw34'], lch, seg_dict['sw34'] // 2,
                                   th_dict['sw34'], m='')
        for _name in ['XS1_l', 'XS2_l', 'XS1_r', 'XS2_r']:
            self.design_transistor(_name, w_dict['sw12'], lch, seg_dict['sw12'] // 2,
                                   th_dict['sw12'], m='')

        # dummy
        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')

        if sym_clk:
            # change clocks on XS2 and XS3
            for inst in ['XS2_l', 'XS2_r']:
                self.reconnect_instance_terminal(inst, 'G', 'RSTD')
            self.reconnect_instance_terminal('XS2', 'G', 'RSTBD')

            for inst in ['XS3_l', 'XS3_r']:
                self.reconnect_instance_terminal(inst, 'G', 'RST')
            self.reconnect_instance_terminal('XS3', 'G', 'RSTB')
