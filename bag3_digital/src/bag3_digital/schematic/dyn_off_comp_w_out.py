# -*- coding: utf-8 -*-

from typing import Dict, Any, List, Tuple, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_digital__dyn_off_comp_w_out(Module):
    """Module for library bag3_digital cell dyn_off_comp_w_out.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'dyn_off_comp_w_out.yaml')))

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
            comp_params='Parameters for Comparator',
            out_params='Parameters for Output stage',
            remove_pins_list='List of pins to be removed',
            rename_pins_list='List of pins to be renamed',
            extra_pins='Extra pins to be added',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            remove_pins_list=[],
            rename_pins_list=[],
            extra_pins=None,
        )

    def design(self, comp_params: Param, out_params: Param, remove_pins_list: List[str],
               rename_pins_list: List[Tuple[str]], extra_pins: Optional[Dict[str, list]]) -> None:
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
        # Comparator
        self.instances['XCOMP'].design(**comp_params)

        # Output stage
        self.instances['XOUT'].design(**out_params)

        # Pins
        for pin in remove_pins_list:
            self.remove_pin(pin)

        for old_pin, new_pin in rename_pins_list:
            self.rename_pin(old_pin, new_pin)

        if extra_pins is not None:
            for pin_name in extra_pins['in']:
                self.add_pin(pin_name, TermType.input)
            for pin_name in extra_pins['out']:
                self.add_pin(pin_name, TermType.output)
