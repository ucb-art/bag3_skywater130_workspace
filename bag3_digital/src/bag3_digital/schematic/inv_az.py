# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param

from pybag.enum import TermType


# noinspection PyPep8Naming
class bag3_digital__inv_az(Module):
    """Module for library bag3_digital cell inv_az.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'inv_az.yaml')))

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
            inv_params='Parameters for Inverter',
            samp_params='Parameters for sampler',
            switch_params='Parameters for auto zeroing switch',
            ideal_switch='True to use ideal switch',
            extra_pins='Extra pins to be added',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            ideal_switch=False,
            switch_params=None,
            extra_pins=None,
        )

    def design(self, inv_params: Param, samp_params: Param, switch_params: Optional[Param],
               ideal_switch: bool, extra_pins: Dict[str, list]) -> None:
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
        self.instances['XINV'].design(**inv_params)

        samp_params = samp_params.copy(append=dict(ideal_switch=ideal_switch))
        self.instances['XSAMP'].design(**samp_params)

        if ideal_switch:
            self.replace_with_ideal_switch('XSaz')
        else:
            self.design_transistor('XSaz', **switch_params)

        if extra_pins is not None:
            for pin_name in extra_pins['in']:
                self.add_pin(pin_name, TermType.input)
            for pin_name in extra_pins['out']:
                self.add_pin(pin_name, TermType.output)
