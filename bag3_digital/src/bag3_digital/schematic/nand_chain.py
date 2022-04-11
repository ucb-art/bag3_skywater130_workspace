# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param, ImmutableList


# noinspection PyPep8Naming
class bag3_digital__nand_chain(Module):
    """Module for library bag3_digital cell nand_chain.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'nand_chain.yaml')))

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
            nand_params='List of NAND parameters.',
            export_pins='True to export simulation pins.',
            dual_output='True to export complementary outputs  Ignored if only one stage.',
            close2supply='True to chain input close to supply.',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            export_pins=False,
            dual_output=False,
            close2supply=True,
        )

    def design(self, nand_params: ImmutableList[Param], export_pins: bool, dual_output: bool,
               close2supply: bool) -> None:
        num = len(nand_params)
        if num < 1:
            raise ValueError('Cannot have 0 gates.')
        if export_pins and dual_output:
            raise ValueError("oops! export_pins and dual_output cannot be True at the same time, "                             
                             "check nand_chain's schematic generator")
        in_name = 'VDD,in' if close2supply else 'in,VDD'
        if num == 1:
            self.instances['XNAND'].design(**nand_params[0])
            self.remove_pin('out')
            self.reconnect_instance_terminal('XNAND', 'in<1:0>', in_name)
        else:
            if num == 2:
                pin_last2 = 'outb'
                if export_pins:
                    pin_last2 = 'mid'
                    self.rename_pin('outb', pin_last2)
                elif not dual_output:
                    self.remove_pin('outb')

                in_name2 = f'VDD,{pin_last2}' if close2supply else f'{pin_last2},VDD'

                inst_term_list = [('XNAND0', [('in<1:0>', in_name), ('out', pin_last2)]),
                                  ('XNAND1', [('in<1:0>', in_name2), ('out', 'out')])]
            else:
                pin_last2 = 'outb' if num % 2 == 0 else 'out'
                if export_pins:
                    pin_last2 = f'mid<{num - 2}>'
                    self.rename_pin('outb' if num % 2 == 0 else 'out', f'mid<{num - 2}:0>')
                elif not dual_output:
                    self.remove_pin('outb' if num % 2 == 0 else 'out')

                inst_term_list = []
                in_name2 = f'VDD,{pin_last2}' if close2supply else f'{pin_last2},VDD'

                for idx in range(num):
                    if idx == 0:
                        term = [('in<1:0>', in_name), ('out', 'mid<0>')]
                    elif idx == num - 1:
                        term = [('in<1:0>', in_name2), ('out', 'out' if num % 2 == 0 else 'outb')]
                    elif idx == num - 2:
                        in_name3 = f'VDD,mid<{idx - 1}>' if close2supply else f'mid<{idx - 1}>,VDD'
                        term = [('in<1:0>', in_name3), ('out', pin_last2)]
                    else:
                        in_name3 = f'VDD,mid<{idx - 1}>' if close2supply else f'mid<{idx - 1}>,VDD'
                        term = [('in<1:0>', in_name3), ('out', f'mid<{idx}>')]
                    inst_term_list.append((f'XNAND{idx}', term))

            self.array_instance('XNAND', inst_term_list=inst_term_list)
            for idx in range(num):
                self.instances[inst_term_list[idx][0]].design(**nand_params[idx])
