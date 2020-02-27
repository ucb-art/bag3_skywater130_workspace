# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__DynOffCompSB_coreN_split(Module):
    """Module for library bag3_digital cell DynOffCompSB_coreN_split.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'DynOffCompSB_coreN_split.yaml')))

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
            ideal_switch='True to use ideal switches.',
            nmos_switch='True to replace pmos switches with nmos switches',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            dum_info=None,
            ideal_switch=False,
            nmos_switch=False,
        )

    def design(self, seg_dict: Dict[str, int], lch: int, w_dict: Dict[str, int],
               th_dict: Dict[str, str], dum_info: List[Tuple[Any]], ideal_switch: bool,
               nmos_switch: bool) -> None:
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

        # nch devices + power gate
        for _name, _str in zip(['XM1', 'XM2', 'XMc1', 'XMc2', 'XPOW'],
                               ['gm2n', 'gm2n', 'gm1', 'gm1', 'pow']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        # pch devices + pch switches
        for _name, _str in zip(['XM3', 'XM4', 'XM1p', 'XM2p'], ['gm2p', 'gm2p', 'gm2p2', 'gm2p2']):
            self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str], m='')

        # dummy
        self.design_dummy_transistors(dum_info, 'XDUM', 'VDD', 'VSS')

        # switches
        if ideal_switch:
            for _name in ['XS1', 'XS2', 'XS3', 'XS4', 'XSc1', 'XSc2']:
                self.replace_with_ideal_switch(_name)
        elif nmos_switch:
            for _name, _str in zip(['XS1', 'XS2', 'XSc1', 'XSc2'],
                                   ['sw12', 'sw12', 'swc12', 'swc12']):
                self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str],
                                       m='', g='RST', mos_type='nch')
            for _name, _str in zip(['XS3', 'XS4'], ['sw34', 'sw34']):
                self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str],
                                       m='', g='RSTD', mos_type='nch')
            self.rename_pin('RSTB', 'RST')
            self.rename_pin('RSTBD', 'RSTD')
        else:
            for _name, _str in zip(['XS1', 'XS2', 'XS3', 'XS4', 'XSc1', 'XSc2'],
                                   ['sw12', 'sw12', 'sw34', 'sw34', 'swc12', 'swc12']):
                self.design_transistor(_name, w_dict[_str], lch, seg_dict[_str], th_dict[_str],
                                       m='')
