# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_digital__wrapper_digital_db_cg(Module):
    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'wrapper_digital_db_cg.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            dut_type='Type of device under test, possible values: "nch" (default), "pch".',
            dut_lib='DUT library name.',
            dut_cell='DUT cell name.',
            dut_conns='DUT connections dictionary',
            source_load_params='Source load params for keeping wrapper interface common',
            tx_params='Transistor parameters',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(source_load_params=None, dut_conns=None)

    def design(self, dut_type: str, dut_lib: str, dut_cell: str,
               dut_conns: Optional[Dict[str, Any]], source_load_params: Optional[Dict[str, Any]],
               tx_params: Dict[str, Any]) -> None:
        # replacing instance master statically
        self.replace_instance_master('XDUT', dut_lib, dut_cell, static=True)

        if dut_type == 'nch':
            default_dut_conns = dict(d='out', g='in', s='VSS', b='VSS')
        else:
            default_dut_conns = dict(d='out', g='in', s='VDD', b='VDD')

        if dut_conns:
            default_dut_conns.update(dut_conns)

        for term, net in default_dut_conns.items():
            self.reconnect_instance_terminal('XDUT', term, net)

        self.design_sources_and_loads(source_load_params)

        if "mos_type" in tx_params:
            raise ValueError('mos_type should not be given in tx_params')

        if dut_type == 'nch':
            self.design_transistor('XTX', **tx_params)
        else:
            if any(pin in tx_params for pin in ['s', 'b']):
                raise ValueError('when dut_type = "pch" both s and b should be defined in '
                                 'tx_params. They will be inferred')

            self.design_transistor('XTX', mos_type='nch', s='VSS', b='VSS', **tx_params)
