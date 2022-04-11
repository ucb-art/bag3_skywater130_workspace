# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__col_decoder(Module):
    """Module for library bag3_digital cell decoder.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'col_decoder.yaml'))

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
            nbits='number of bits',
            master_params='parameters of inv, nand, nor (they are all similar)',
            row_masters='list of type of masters from bottom to top',
            is_row_decoder='True if its a row decoder'
        )

    @classmethod
    def get_default_param_values(cls):  # type: () -> Dict[str, Any]:
        return dict(is_row_decoder=False)

    def design(self, nbits: int, master_params: Param, row_masters: List[str],
               is_row_decoder: bool = False) -> None:
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
        if nbits == 1:
            raise ValueError('1 bit is not supported yet!! cds_tru is needed')

        n_cols = 2 ** nbits
        out_suffix = '<%d:0>' % (n_cols - 1)
        input_suffix = '<%d:0>' % (nbits - 1)
        sch_base_names = ['Xinv', 'Xnand', 'Xnor']

        # rename pins
        self.rename_pin('in', 'in' + input_suffix)
        self.rename_pin('inb', 'inb' + input_suffix)
        self.rename_pin('en', 'en' + out_suffix)
        self.rename_pin('enb', 'enb' + out_suffix)

        # check if we shouldn't put any of the masters for some reason
        for base_name in ['inv', 'nand', 'nor']:
            if base_name not in row_masters:
                self.delete_instance('X'+base_name)
                sch_base_names.remove('X'+base_name)

        # prepare name_list and term_list data structures
        term_list, name_list = {}, {}
        for name in sch_base_names:
            term_list[name] = []
            name_list[name] = []

        output_list: List[List[Tuple]] = [[() for _ in range(n_cols)]
                                          for _ in range(len(row_masters))]

        # create instances from bottom to top
        for r, master in enumerate(row_masters):
            for c in range(n_cols):
                if master == 'inv':
                    out = 'out_{row}_{col}'.format(row=r, col=c)
                    term = {
                        'in': output_list[r-1][c][0],
                        'out': out,
                        'VDD': 'VDD',
                        'VSS': 'VSS',
                    }
                    output_list[r][c] = (out,)
                    name = 'Xinv_{row}_{col}'.format(row=r, col=c)
                    term_list['Xinv'].append(term)
                    name_list['Xinv'].append(name)

                    # params_list.append(master_params.copy())

                elif master == 'nand':
                    out = 'out_{row}_{col}'.format(row=r, col=c)
                    if r == 0:
                        repeat = c // (2 ** (nbits - 2))
                        base_0 = 'inb' if repeat // 2 else 'in'
                        base_1 = 'inb' if repeat % 2 else 'in'
                        term = {
                            'in0': base_0 + '<{}>'.format(nbits - 1),
                            'in1': base_1 + '<{}>'.format(nbits - 2),
                            'VSS': 'VSS',
                            'VDD': 'VDD',
                            'out': out,
                        }
                    else:
                        term = {
                            'in0': output_list[r-1][c][0],
                            'in1': output_list[r-1][c][1],
                            'VSS': 'VSS',
                            'VDD': 'VDD',
                            'out': out,
                        }

                    output_list[r][c] = (out,)
                    name = 'Xnand_{row}_{col}'.format(row=r, col=c)
                    term_list['Xnand'].append(term)
                    name_list['Xnand'].append(name)

                elif master == 'nor':
                    out = 'out_{row}_{col}'.format(row=r, col=c)
                    if r == 0:
                        repeat = c // (2 ** (nbits - 2))
                        base_0 = 'inb' if repeat // 2 else 'in'
                        base_1 = 'inb' if repeat % 2 else 'in'
                        term = {
                            'in0': base_0 + '<{}>'.format(nbits - 1),
                            'in1': base_1 + '<{}>'.format(nbits - 2),
                            'VSS': 'VSS',
                            'VDD': 'VDD',
                            'out': out,
                        }
                    else:
                        term = {
                            'in0': output_list[r - 1][c][0],
                            'in1': output_list[r - 1][c][1],
                            'VSS': 'VSS',
                            'VDD': 'VDD',
                            'out': out,
                        }

                    output_list[r][c] = (out,)
                    name = 'Xnor_{row}_{col}'.format(row=r, col=c)
                    term_list['Xnor'].append(term)
                    name_list['Xnor'].append(name)

                else:
                    raise ValueError('invalid instance in schematic')

                # For rows below the last two rows we have at least one external input
                if r < len(row_masters) - 2:
                    repeat = c // (2 ** (nbits - 3 - r))
                    base = 'inb' if repeat % 2 else 'in'
                    if master == 'nand':
                        base = 'in' if base == 'inb' else 'inb'

                    if is_row_decoder:
                        if c % 2 != 0:
                            output_list[r][c] = (base + '<{}>'.format(nbits - 3 - r),) + \
                                                output_list[r][c]
                        else:
                            output_list[r][c] += (base + '<{}>'.format(nbits - 3 - r),)
                    else:
                        output_list[r][c] += (base + '<{}>'.format(nbits - 3 - r),)

        # rename the output_list names to en or enb
        for c in range(n_cols):
            if nbits % 2:
                # if odd output of inverter is enb
                term_list['Xinv'][c]['out'] = 'enb<%d>' % c
                term_list['Xinv'][c]['in'] = 'en<%d>' % c
                term_list['Xnor'][(nbits // 2 - 1) * n_cols + c]['out'] = 'en<%d>' % c
            else:
                # if even output of inverter is en
                term_list['Xinv'][c]['out'] = 'en<%d>' % c
                term_list['Xinv'][c]['in'] = 'enb<%d>' % c
                term_list['Xnand'][(nbits // 2 - 1) * n_cols + c]['out'] = 'enb<%d>' % c

        for master in term_list.keys():
            self.array_instance(master, name_list[master], term_list[master])

        for name, inst in self.instances.items():
            if name.startswith('Xinv'):
                params = master_params.copy()
            else:
                params = master_params.copy(append={'seg': master_params['seg'] // 2})
            inst.design(**params)
