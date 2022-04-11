from typing import Dict, Any, Tuple

from bag3_testbenches.measurement.mos.db import (
    OutputDriverTBManager
)

from bag.simulation.core import MeasurementManager
from bag.simulation.data import SimData

import pprint
import matplotlib.pyplot as plt


class OutputDriverMeasManager(MeasurementManager):

    def __init__(self, *args, **kwargs):
        super(OutputDriverMeasManager, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> str:
        tbs = self.specs['testbenches']
        if len(tbs) > 1:
            raise ValueError('OutputDriverMeasManager cannot have more than one tb')
        return list(tbs.keys())[0]

    def process_output(self,
                       state: str,
                       data: SimData,
                       tb_manager: OutputDriverTBManager,
                       ) -> Tuple[bool, str, Dict[str, Any]]:

        tper = tb_manager.specs['sim_params']['tper']
        cout = tb_manager.specs['sim_params']['cout']
        vdd = tb_manager.specs['sim_params']['vdd']
        if state == 'output_res':
            res_data = tb_manager.get_output_res(data, tper, cout, vdd=vdd)

            results = dict(
                pullup_res=res_data['pu'],
                pulldown_res=res_data['pd'],
            )

        elif state == 'tr_tf':
            trf_data = tb_manager.get_tr_tf(data, tper, vdd_output=vdd)
            # trf_data = tb_manager.get_tr_tf(data, tper, clk_trf, vdd=vdd, out_key='midp')
            results = dict(
                tr=trf_data['tr'],
                tf=trf_data['tf'],
            )
            print(f'tr: {results["tr"]}')
            print(f'tf: {results["tf"]}')

        elif state == 'tdelay':
            vdd_input = tb_manager.specs['sim_params']['vdd_core']
            td_in_midp = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                              in_key='in',  out_key='midp')
            td_midp_midn = tb_manager.get_delay(data, tper, vdd_input=vdd, vdd_output=vdd,
                                                in_key='midp', out_key='midn')
            td_in_midn = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                              in_key='in', out_key='midn')
            td_in_inb = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd_input,
                                             in_key='in', out_key='inbbuf')
            td_inb_midn = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                               in_key='inbbuf', out_key='midn')
            # td_in_midn2 = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
            #                                    in_key='in', out_key='midn')

            # print('l2h in -> midn (1)')
            # print(f'in -> midp: {td_in_midp["tdl2h"]:6.4e}, '
            #       f'midp -> midn: {td_midp_midn["tdh2l"]:6.4g}')
            # print(f'in -> midn (1): {td_in_midn["tdl2h"]:6.4g}')
            #
            # print('h2l in -> midn (1)')
            # print(f'in -> midp: {td_in_midp["tdh2l"]:6.4e}, '
            #       f'midp -> midn: {td_midp_midn["tdl2h"]:6.4g}')
            # print(f'in -> midn (1): {td_in_midn["tdh2l"]:6.4g}')
            #
            # print('l2h in -> midn (2)')
            # print(f'in -> inb: {td_in_inb["tdl2h"]:6.4e}, '
            #       f'inb -> midn: {td_inb_midn["tdh2l"]:6.4g}')
            # print(f'in -> midn (2): {td_in_midn2["tdl2h"]:6.4g}')
            #
            # print('h2l in -> midn (2)')
            # print(f'in -> inb: {td_in_inb["tdh2l"]:6.4e}, '
            #       f'inb -> midn: {td_inb_midn["tdl2h"]:6.4g}')
            # print(f'in -> midn (2): {td_in_midn2["tdh2l"]:6.4g}')

            plt.clf()
            plt.subplot(211)
            # plt.plot(data['time'], data['inbuf'][0], label='inbuf')
            plt.plot(data['time'], data['inbbuf'][0], label='inbbuf')
            plt.plot(data['time'], data['in'][0], label='in')
            plt.legend()
            plt.subplot(212)
            plt.plot(data['time'], data['inbbuf'][0], label='inbbuf')
            plt.plot(data['time'], data['in'][0], label='in')
            plt.plot(data['time'], data['midn'][0], label='midn')
            plt.plot(data['time'], data['midp'][0], label='midp')
            plt.legend()

            results = dict(
                ttop1=td_in_midp["tdl2h"],
                ttop2=td_midp_midn["tdh2l"],
                tbot1=td_in_inb["tdh2l"],
                tbot2=td_inb_midn["tdl2h"],
                ttop=td_in_midn["tdl2h"],
                tbot=td_in_midn["tdh2l"],
            )

            if self.specs['plot_figs']:
                pprint.pprint(results)
                plt.show()

        else:
            raise KeyError('Invalid state!')

        return True, '', results
