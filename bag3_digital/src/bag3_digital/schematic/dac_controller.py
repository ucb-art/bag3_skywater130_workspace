# -*- coding: utf-8 -*-

from typing import Dict, Any

import os
import math
import pkg_resources

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__dac_controller(Module):
    """Module for library bag3_digital cell dac_controller.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                os.path.join('netlist_info',
                                                             'dac_controller.yaml'))

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
            n_rows='number of resistor rows',
            n_cols='number of resistor columns',
            inv_buffer_params='inverter buffer params',
            col_decoder_params='column decoder params',
            row_decoder_params='row decoder params',
            passgate_arr_params='passgate array params'
        )

    def design(self,
               n_rows: int, n_cols: int,
               inv_buffer_params: Param,
               col_decoder_params: Param,
               row_decoder_params: Param,
               passgate_arr_params: Param) -> None:
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

        n_inputs = n_rows * n_cols
        n_selects = int(math.log2(n_inputs))
        n_col_bits = int(math.log2(n_cols))
        n_row_bits = int(math.log2(n_rows))

        print(n_selects)

        input_suffix = f'<{n_inputs-1}:0>'
        select_suffix = f'<{n_selects-1}:0>'
        c_en_suffix = f'<{n_cols-1}:0>'
        c_sel_suffix = f'<{n_col_bits-1}:0>'
        r_en_suffix = f'<{n_rows-1}:0>'
        r_sel_suffix = f'<{n_selects-1}:{n_col_bits}>'
        r_sel_suffix2 = f'<{n_row_bits-1}:0>'

        # import pdb
        # pdb.set_trace()
        # rename pins
        self.rename_pin('in', 'in' + input_suffix)
        self.rename_pin('sel', 'sel' + select_suffix)

        # connect inverter buffer terminals
        self.instances['Xinv_buffer'].design(**inv_buffer_params)
        self.reconnect_instance_terminal('Xinv_buffer', 'sel' + select_suffix,
                                         'sel' + select_suffix)
        self.reconnect_instance_terminal('Xinv_buffer', 'selb' + select_suffix,
                                         'selb' + select_suffix)

        # connect column decoder pins
        self.instances['Xcol_decoder'].design(**col_decoder_params)
        self.reconnect_instance_terminal('Xcol_decoder', 'en' + c_en_suffix,
                                         'c_en' + c_en_suffix)
        self.reconnect_instance_terminal('Xcol_decoder', 'enb' + c_en_suffix,
                                         'c_enb' + c_en_suffix)
        self.reconnect_instance_terminal('Xcol_decoder', 'in' + c_sel_suffix,
                                         'sel' + c_sel_suffix)
        self.reconnect_instance_terminal('Xcol_decoder', 'inb' + c_sel_suffix,
                                         'selb' + c_sel_suffix)

        # connect row decoder pins
        self.instances['Xrow_decoder'].design(**row_decoder_params)
        self.reconnect_instance_terminal('Xrow_decoder', 'en' + r_en_suffix,
                                         'r_en' + r_en_suffix)
        self.reconnect_instance_terminal('Xrow_decoder', 'enb' + r_en_suffix,
                                         'r_enb' + r_en_suffix)
        self.reconnect_instance_terminal('Xrow_decoder', 'in' + r_sel_suffix2,
                                         'sel' + r_sel_suffix)
        self.reconnect_instance_terminal('Xrow_decoder', 'inb' + r_sel_suffix2,
                                         'selb' + r_sel_suffix)
        # connect passgate array pins
        self.instances['Xpass_gate'].design(**passgate_arr_params)
        self.reconnect_instance_terminal('Xpass_gate', 'in' + input_suffix,
                                         'in' + input_suffix)
        self.reconnect_instance_terminal('Xpass_gate', 'c_en' + c_en_suffix,
                                         'c_en' + c_en_suffix)
        self.reconnect_instance_terminal('Xpass_gate', 'c_enb' + c_en_suffix,
                                         'c_enb' + c_en_suffix)
        self.reconnect_instance_terminal('Xpass_gate', 'r_en' + r_en_suffix,
                                         'r_en' + r_en_suffix)
        self.reconnect_instance_terminal('Xpass_gate', 'r_enb' + r_en_suffix,
                                         'r_enb' + r_en_suffix)