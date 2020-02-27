# -*- coding: utf-8 -*-

from typing import Dict, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_analog__variablecapchain2(Module):
    """Module for library bag3_analog cell variablecapchain2.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'variablecapchain2.yaml')))

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
            lch='Channel length.',
            p_gates='List of pgate info, from innermost to outermost',
            n_gates='List of pgate info, from innermost to outermost',
            p_caps='List of pcap info, from innermost to outermost',
            n_caps='List of ncap info, from innermost to outermost',
        )

    def design(self, lch, p_gates, n_gates, p_caps, n_caps) -> None:
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
        # design capacitors

        assert len(n_caps) == len(p_caps)
        assert len(n_gates) == len(p_gates)
        assert len(n_gates) == len(n_caps)

        self._design_chain_helper(lch, 'XNCAP', n_caps, 'XNGATE', n_gates, 'VSS', 'en', 'mid')
        self._design_chain_helper(lch, 'XPCAP', p_caps, 'XPGATE', p_gates, 'VDD', 'enb', 'midb')

    def _design_chain_helper(self, lch, cap_name, cap_list, gate_name, gate_list,
                             supply, enable, mid):
        if not cap_list:
            self.delete_instance(cap_name)
            self.delete_instance(gate_name)
            self.remove_pin(supply)
            return

        num_caps = len(gate_list)
        if num_caps == 1:
            self.design(cap_name, cap_list[0])
            tx_info = gate_list[0]
            self.design_transistor(gate_name, tx_info['w'], lch, tx_info['seg'], tx_info['th'])
        else:
            cap_name_list = [f'{cap_name}{x}' for x in range(num_caps)]
            self.array_instance(cap_name, inst_name_list=cap_name_list)
            gate_name_list = [f'{gate_name}{x}' for x in range(num_caps)]
            self.array_instance(gate_name, inst_name_list=gate_name_list)
            for i in range(num_caps):
                cap_inst_name = f'{cap_name}{i}'
                cap_info = cap_list[i]
                self.instances[cap_inst_name].design(**cap_info)

                gate_inst_name = f'{gate_name}{i}'
                tx_info = gate_list[i]
                self.design_transistor(gate_inst_name, tx_info['w'], lch, tx_info['seg'],
                                       tx_info['th'])

                if i > 0:
                    self.reconnect_instance_terminal(gate_inst_name, 'D', f'{mid}{i-1}')
                self.reconnect_instance_terminal(gate_inst_name, 'G', f'{enable}<{i}>')
                self.reconnect_instance_terminal(gate_inst_name, 'S', f'{mid}{i}')

                self.reconnect_instance_terminal(cap_inst_name, 'in', f'{mid}{i}')
            self.rename_pin(enable, f'{enable}<0:{num_caps-1}>')