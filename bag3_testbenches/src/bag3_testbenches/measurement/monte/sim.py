# SPDX-License-Identifier: Apache-2.0
# Copyright 2019 Blue Cheetah Analog Design Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This package contains measurement class for mismatch parameters of transistors."""

from typing import Tuple, Dict, Any, Sequence, Union, List, cast, Optional, Mapping

from pathlib import Path

import numpy as np
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasInfo, MeasurementManager, TestbenchManager
from bag3_testbenches.measurement.dc.base import DCTB


class MOSMismatchMM(MeasurementManager):
    """ This class measures Avt and Abeta of transistors """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tbm_specs = None
        self._tbm_info: Optional[Tuple[DCTB, Mapping[str, Any]]] = None
        self.nfin = 0
        self.nmos = None
        self.vdd = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        ib_start: float = specs["ib_start"]
        ib_stop: float = specs["ib_stop"]
        num_sweep_pts: int = specs.get("num_sweep_pts", 10)
        vcvs_gain: float = specs.get('vcvs_gain', 1e6)
        inp_pin: str = specs['inp_pin']
        inn_pin: str = specs['inn_pin']
        outp_pin: str = specs['outp_pin']
        outn_pin: str = specs['outn_pin']
        src_list: Optional[Sequence[Dict[str, Any]]] = specs.get('src_list', [])
        sim_params: Dict[str, Any] = specs['sim_params']
        sim_envs: Sequence[str] = specs['sim_envs']

        dut_params = dut.sch_master.params
        self.nfin = dut_params['w'] * dut_params['seg']
        nmos = dut_params['mos_type'] == 'nch'
        self.nmos = nmos
        sup = 'VSS' if nmos else 'VDD'
        sup_bar = 'VDD' if nmos else 'VSS'
        vdd = self.specs['sup_values']['VDD']
        self.vdd = vdd
        value_dict = dict(egain=vcvs_gain, minm=0) if nmos else dict(egain=vcvs_gain, maxm=vdd)
        conn_p = dict(MINUS=outp_pin, PLUS=sup_bar) if nmos else dict(PLUS=outn_pin, MINUS=sup_bar)
        conn_n = dict(MINUS=outn_pin, PLUS=sup_bar) if nmos else dict(PLUS=outp_pin, MINUS=sup_bar)

        sweep_options = dict(type='LINEAR', start=ib_start, stop=ib_stop, num=num_sweep_pts)
        dut_pins = list(dut.sch_master.pins.keys())
        pwr_domain = {pin: ('VSS', 'VDD') for pin in dut_pins}
        src_list.append(dict(type='vcvs', lib='analogLib', value=value_dict,
                             conns={'PLUS': inp_pin, 'MINUS': sup, 'NC+': outn_pin, 'NC-': 'vcm'}))
        src_list.append(dict(type='vcvs', lib='analogLib', value=value_dict,
                             conns={'PLUS': inn_pin, 'MINUS': sup, 'NC+': outp_pin, 'NC-': 'vcm'}))
        src_list.append(dict(type='vcvs', lib='analogLib', value=1.0,
                             conns={'PLUS': 'vin_diff', 'MINUS': 'VSS', 'NC+': inp_pin, 'NC-': inn_pin}))
        src_list.append(dict(type='idc', lib='analogLib', value='ibias', conns=conn_p))
        src_list.append(dict(type='idc', lib='analogLib', value='ibias', conns=conn_n))
        tbm_specs = dict(
            sweep_var='ibias',
            sweep_options=sweep_options,
            dut_pins=dut_pins,
            src_list=specs['src_list'],
            load_list=[],
            pwr_domain=pwr_domain,
            sup_values=specs['sup_values'],
            pin_values=specs['pin_values'],
            sim_params=sim_params,
            sim_envs=sim_envs,
        )
        tb_params = dict()
        self._tbm_specs = tbm_specs
        self._tbm_info = cast(DCTB, sim_db.make_tbm(DCTB, tbm_specs)), tb_params
        return False, MeasInfo('sim_nom', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]], MeasurementManager], bool]:
        if cur_info.state == 'sim_nom':
            return self._tbm_info, True
        else:
            mc_params: Mapping[str, Any] = self.specs.get('monte_carlo_params', None)
            tbm_specs = self._tbm_specs.copy()
            tbm_specs['monte_carlo_params'] = mc_params
            tbm_info = cast(DCTB, sim_db.make_tbm(DCTB, tbm_specs)), self._tbm_info[1]
            return tbm_info, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        if cur_info.state == 'sim_nom':
            ibias = sim_results.data['ibias']
            vgs = sim_results.data['inp']
            if not self.nmos:
                vgs = self.vdd-vgs
            gm = np.diff(ibias)/np.diff(vgs)
            vstar = 2*ibias/np.average(gm)
            return False, MeasInfo('sim_mc', dict(vstar=vstar))
        elif cur_info.state == 'sim_mc':
            vstar_arr = cur_info.prev_results['vstar']
            voffset = sim_results.data['vin_diff']
            sigma_offset = np.var(voffset, axis=-2)[0]
            n = len(sigma_offset)
            amat = np.vstack((np.ones(n), vstar_arr**2/4))
            x = np.linalg.lstsq(amat.T, sigma_offset)[0]
            avt = x[0]
            abeta = x[1]
            avt_per_fin = np.sqrt(avt * self.nfin)
            abeta_per_fin = np.sqrt(abeta * self.nfin)
            if self.specs.get('plot', False):
                plt.plot(vstar_arr**2, avt + vstar_arr**2/4*abeta)
                plt.scatter(vstar_arr**2, sigma_offset)
                plt.show()
            return True, MeasInfo('Done', dict(vstar=vstar_arr,
                                               avt_fin=avt_per_fin, abeta_fin=abeta_per_fin))
