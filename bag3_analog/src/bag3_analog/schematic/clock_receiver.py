# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__clock_receiver(Module):
    """Module for library bag3_analog cell clock_receiver.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'clock_receiver.yaml')))

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
            esd_top_params='Parameters for ESD diodes to supply',
            esd_bot_params='Parameters for ESD diodes to gnd',
            core_params='Parameters for self biased diffamp',
        )

    def design(self, esd_top_params: Param, esd_bot_params: Param, core_params: Param) -> None:
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
        self.instances['XDIFF_BUF'].design(**core_params)

        self.instances['XESD_P_TOP'].design(**esd_top_params)
        self.instances['XESD_N_TOP'].design(**esd_top_params)

        self.instances['XESD_P_BOT'].design(**esd_bot_params)
        self.instances['XESD_N_BOT'].design(**esd_bot_params)