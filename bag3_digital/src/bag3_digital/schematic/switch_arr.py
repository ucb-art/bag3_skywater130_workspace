# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__switch_arr(Module):
    """Module for library bag3_digital cell switch_arr.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'switch_arr.yaml'))

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
            n_cols='number of columns',
            n_rows='number of rows',
            passgate_params='passgate parameters',
        )

    def design(self, n_cols: int, n_rows: int, passgate_params: Dict[str, Any]) -> None:
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

        if n_cols == 1:
            col_suffix = '<0>'
        else:
            col_suffix = '<%d:0>' % (n_cols - 1)

        if n_rows == 1:
            row_suffix = '<0>'
        else:
            row_suffix = '<%d:0>' % (n_rows - 1)

        in_suffix = '<%d:0>' % (n_cols * n_rows - 1)

        name_list, term_list = [],  []

        # rename pins
        self.rename_pin('in', 'in' + in_suffix)
        self.rename_pin('c_en', 'c_en' + col_suffix)
        self.rename_pin('c_enb', 'c_enb' + col_suffix)
        self.rename_pin('r_en', 'r_en' + row_suffix)
        self.rename_pin('r_enb', 'r_enb' + row_suffix)

        # create column instances
        for r in range(n_rows):
            for c in range(n_cols):
                in_number = r * n_cols + c

                term_list.append({
                    'VDD': 'VDD',
                    'VSS': 'VSS',
                    'en': 'c_en<%d>' % c,
                    'enb': 'c_enb<%d>' % c,
                    's': 'in<%d>' % in_number,
                    'd': 'outm<%d>' % r,
                })
                name_list.append('XSc_{row}_{col}'.format(row=r, col=c))

        # create row instances
        for r in range(n_rows):

            term_list.append({
                'VDD': 'VDD',
                'VSS': 'VSS',
                'en': 'r_en<%d>' % r,
                'enb': 'r_enb<%d>' % r,
                's': 'outm<%d>' % r,
                'd': 'out'
            })
            name_list.append('XSr_{}'.format(r))

        self.array_instance('XS', name_list, term_list)

        for inst in self.instances.values():
            inst.design(**passgate_params)
