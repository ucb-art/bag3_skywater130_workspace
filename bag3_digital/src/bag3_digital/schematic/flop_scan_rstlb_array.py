# -*- coding: utf-8 -*-

from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__flop_scan_rstlb_array(Module):
    """Module for library bag3_digital cell flop_scan_rstlb_array.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'flop_scan_rstlb_array.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        return dict(
            lch='channel length.',
            seg_dict='segments dictionary.',
            w_dict='width dictionary.',
            th_p='PMOS threshold.',
            th_n='NMOS threshold.',
            num='number of flops in the array.'
        )

    def design(self, lch: int, seg_dict: Mapping[str, int], w_dict: Mapping[str, int],
               th_p: str, th_n: str, num: int) -> None:
        if num < 2:
            raise ValueError('num must be > 2.')

        old_suf = '<1:0>'
        dut_name = 'XFLOP' + old_suf
        self.instances[dut_name].design(lch=lch, seg_dict=seg_dict, w_dict=w_dict, th_p=th_p,
                                        th_n=th_n)
        if num > 2:
            new_suf = f'<{num - 1}:0>'
            dut_conns = {}
            for basename in ['in', 'out']:
                dut_conns[basename] = new_pin = basename + new_suf
                self.rename_pin(basename + old_suf, new_pin)

            dut_conns['scan_in'] = f'out<{num - 2}:0>,scan_in'
            self.rename_instance(dut_name, 'XFLOP + new_suf', dut_conns.items())
