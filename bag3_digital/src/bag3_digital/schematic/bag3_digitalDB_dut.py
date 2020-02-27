# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional, List

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param
from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_digital__bag3_digitalDB_dut(Module):
    """Module for library bag3_digital cell bag3_digitalDB_dut.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'bag3_digitalDB_dut.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            init_seg='initial nseg used for the first inverter',
            lch='channel length in resolution units.',
            w_p='pmos width, in number of fins or resolution units.',
            pn_ratio='w_n is computed using this ratio',
            fanout='the fanout between stages',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            mode='This controls the cg/cd/R measuring mode',
            in_gate_numbers='List of input indices in case of stack zero is the bottom one',
            vdd_input='True if input supply is different from output supply (level shifter),'
                      'in this case VDD_in and VDD will be the pin names',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            in_gate_numbers=None,
            vdd_input=False,
        )

    def design(self, init_seg: int, lch: int, w_p: int, pn_ratio: float, fanout: float, th_p: str,
               th_n: str, stack_p: int, stack_n: int, mode: str,
               in_gate_numbers: Optional[List[int]] = None,
               vdd_input: bool = False) -> None:

        w_n = int(w_p / pn_ratio)
        inv1_params = dict(
            seg=init_seg,
            lch=lch,
            w_p=w_p,
            w_n=w_n,
            th_n=th_n,
            th_p=th_p,
            stack_n=stack_n,
            stack_p=stack_p,
        )
        inv2_params = inv1_params.copy()
        inv2_params['seg'] *= fanout
        inv_out_params = inv1_params.copy()
        seg = int(inv1_params['seg'] * 2 * (fanout ** 2))

        if mode.startswith('cg'):

            # supporting stacks with non common gates
            if mode == 'cgp':
                # new_params = dict(pin_gate_numbers=in_gate_numbers)
                if not in_gate_numbers:
                    pgnet = 'in_buf'
                else:
                    pgnet = ['VSS'] * stack_p
                    for i in in_gate_numbers:
                        pgnet[i] = 'in_buf'
                ngnet = 'vcvs_out'
            else:
                # new_params = dict(nin_gate_numbers=in_gate_numbers)
                if not in_gate_numbers:
                    ngnet = 'in_buf'
                else:
                    ngnet = ['VDD_in' if vdd_input else 'VDD'] * stack_n
                    for i in in_gate_numbers:
                        ngnet[i] = 'in_buf'
                pgnet = 'vcvs_out'

            # inv1_params.update(new_params)
            # inv2_params.update(new_params)
            inv_out_params['seg'] = int(inv_out_params['seg'] * 2 * (fanout ** 3))

            xp_terms = dict(
                d='out',
                g=pgnet,
                s='VDD',
                b='VDD',
            )

            xn_terms = dict(
                d='out',
                g=ngnet,
                s='VSS',
                b='VSS',
            )

            self.design_transistor('XP', w_p, lch, seg, th_p, m='midp', stack=stack_p, **xp_terms)
            self.design_transistor('XN', w_n, lch, seg, th_n, m='midn', stack=stack_n, **xn_terms)

            self.reconnect_instance('Xvcvs', [('PLUS', 'vcvs_out'),
                                              ('MINUS', 'VSS'),
                                              ('NC+', 'in_buf'),
                                              ('NC-', 'VSS')])
        elif mode.startswith('cd'):
            inv_out_params['seg'] = int(inv_out_params['seg'] * fanout ** 2)
            self.delete_instance('Xvcvs')
            self.reconnect_instance_terminal('Xinv_out', term_name='in', net_name='in_buf')

            if mode == 'cdp':
                if in_gate_numbers:
                    gnets = ['VSS'] * stack_p
                    for i in in_gate_numbers:
                        gnets[i] = 'VDD'
                else:
                    gnets = 'VDD'
                terms = dict(
                    d='in_buf',
                    g=gnets,
                    s='VDD',
                    b='VDD',
                )
                self.delete_instance('XN')
                self.design_transistor('XP', w_p, lch, seg, th_p, m='midp', stack=stack_p, **terms)
            else:
                if in_gate_numbers:
                    gnets = ['VDD_in' if vdd_input else 'VDD'] * stack_n
                    for i in in_gate_numbers:
                        gnets[i] = 'VSS'
                else:
                    gnets = 'VSS'
                terms = dict(
                    d='in_buf',
                    g=gnets,
                    s='VSS',
                    b='VSS',
                )
                self.delete_instance('XP')
                self.design_transistor('XN', w_n, lch, seg, th_n, m='midn', stack=stack_n, **terms)

        elif mode.startswith('res'):
            seg = int(inv_out_params['seg'] * fanout ** 2)
            inv_out_params['seg'] = int(inv_out_params['seg'] * fanout ** 2)
            self.reconnect_instance('Xinv_out', [('VDD', 'VDD'),
                                                 ('VSS', 'VSS'),
                                                 ('in', 'in_buf'),
                                                 ('out', 'dummy_out')])

            self.reconnect_instance('Vswitch', [('PLUS', 'switch_g'), ('MINUS', 'VSS')])
            self.reconnect_instance('Xvcvs', [('PLUS', 'vcvs_inbuf'),
                                              ('MINUS', 'VSS'),
                                              ('NC+', 'in_buf'),
                                              ('NC-', 'VSS')])
            self.reconnect_instance('Xswitch', [('NC+', 'switch_g' if mode == 'resp' else 'VDD'),
                                                ('NC-', 'VSS' if mode == 'resp' else 'switch_g'),
                                                ('N-', 'out'),
                                                ('N+', 'VSS' if mode == 'resp' else 'VDD')])
            self.array_instance('Xcap', ['Xcload', 'Xcload_dummy'])
            self.reconnect_instance('Xcload', [('PLUS', 'out'), ('MINUS', 'VSS')])
            self.reconnect_instance('Xcload_dummy', [('PLUS', 'dummy_out'), ('MINUS', 'VSS')])
            if mode == 'resp':
                self.delete_instance('XN')
                # xn_terms = dict(
                #     d='out',
                #     g='VSS',
                #     s='VSS',
                #     b='VSS',
                # )
                if in_gate_numbers:
                    gnets = ['VSS'] * stack_p
                    for i in in_gate_numbers:
                        gnets[i] = 'vcvs_inbuf'
                else:
                    gnets = 'vcvs_inbuf'

                xp_terms = dict(
                    d='out',
                    g=gnets,
                    s='VDD',
                    b='VDD',
                )
                self.design_transistor('XP', w_p, lch, seg, th_p, m='midp', stack=stack_p,
                                       **xp_terms)
                # self.design_transistor('XN', w_n, lch, seg, th_n, m='midn', stack=stack_n,
                #                        **xn_terms)
            else:
                self.delete_instance('XP')

                if in_gate_numbers:
                    gnets = ['VDD_in' if vdd_input else 'VDD'] * stack_n
                    for i in in_gate_numbers:
                        gnets[i] = 'vcvs_inbuf'
                else:
                    gnets = 'vcvs_inbuf'

                xn_terms = dict(
                    d='out',
                    g=gnets,
                    s='VSS',
                    b='VSS',
                )
                # xp_terms = dict(
                #     d='out',
                #     g='VDD',
                #     s='VDD',
                #     b='VDD',
                # )
                self.design_transistor('XN', w_n, lch, seg, th_n, m='midn', stack=stack_n,
                                       **xn_terms)
                # self.design_transistor('XP', w_p, lch, seg, th_p, m='midp', stack=stack_p,
                #                        **xp_terms)

        elif mode.startswith('cload'):
            self.delete_instance('Xvcvs')
            self.delete_instance('XP')
            self.delete_instance('XN')
            if mode == 'cloadg':
                self.delete_instance('Xinv_out')
            else:
                self.reconnect_instance_terminal('Xinv_out', term_name='in', net_name='in_buf')
            inv_out_params['seg'] *= fanout ** 2
            self.reconnect_instance('Xcap', [('PLUS', 'in_buf'), ('MINUS', 'VSS')])
        else:
            raise ValueError('Unkown mode for this schematic generator')

        if mode not in ['cloadd', 'cloadg', 'resp', 'resn']:
            self.delete_instance('Xcap')

        if not mode.startswith('res'):
            self.delete_instance('Xswitch')
            self.delete_instance('Vswitch')

        self.instances['Xinv1'].design(**inv1_params)
        self.instances['Xinv2'].design(**inv2_params)
        if 'Xinv_out' in self.instances:
            self.instances['Xinv_out'].design(**inv_out_params)

        if vdd_input:
            self.add_pin('VDD_in', TermType.output)
            if mode in ['cgn', 'cgp', 'cloadg']:
                self.reconnect_instance_terminal('Xinv1', 'VDD', 'VDD_in')
                self.reconnect_instance_terminal('Xinv2', 'VDD', 'VDD_in')



