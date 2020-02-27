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

"""This package contains transient noise measurement class for comparators."""

from typing import Optional, Dict, Any, cast

from pathlib import Path

import scipy.optimize as sciopt
from scipy.stats import norm

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.concurrent.util import GatherHelper
from bag.math import float_to_si_string

from .tran import CompTranTB, OverdriveMM


class OverdriveNoiseMM(OverdriveMM):
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """Setup parallel processing"""
        helper = GatherHelper()

        # Setup parallel simulations and measurements
        corners = self.specs['corners']
        eval_ampl_dict = self.specs['eval_ampl']
        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            for vdd_idx, vdd in enumerate(vdd_list):
                eval_ampl_list = eval_ampl_dict[env]['vals'][vdd_idx]
                for eval_ampl in eval_ampl_list:
                    helper.append(self.meas_cor_vdd_eval(name, sim_dir, sim_db, dut, env, vdd,
                                                         eval_ampl))

        coro_results = await helper.gather_err()
        results = {}

        glob_idx = 0
        keys = ['ones01', 'ones10']

        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            results[env] = dict(vdd=vdd_list)
            results[env].update({key: [[] for _ in range(len(vdd_list))] for key in keys})
            results[env].update({key: [] for key in ['sigma01', 'sigma10']})
            for vdd_idx, vdd in enumerate(vdd_list):
                eval_ampl_list = eval_ampl_dict[env]['vals'][vdd_idx]
                for eval_ampl in eval_ampl_list:
                    for key in keys:
                        results[env][key][vdd_idx].append(coro_results[glob_idx][key])
                    glob_idx += 1
                _, sigma01 = sciopt.curve_fit(norm.cdf, eval_ampl_list,
                                              results[env]['ones01'][vdd_idx], p0=[0, 1])[0]
                results[env]['sigma01'].append(sigma01)
                _, sigma10 = sciopt.curve_fit(norm.cdf, eval_ampl_list,
                                              results[env]['ones10'][vdd_idx], p0=[0, 1])[0]
                results[env]['sigma10'].append(sigma10)

        return results

    async def meas_cor_vdd_eval(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                dut: Optional[DesignInstance], env: str, vdd: float,
                                eval_ampl: float) -> Dict[str, Any]:
        """Run measurement per corner, supply voltage, and evaluation amplitude"""
        tbm_specs = self._get_tbm_specs(env, vdd)
        sim_dir = sim_dir / env / float_to_si_string(vdd) / float_to_si_string(eval_ampl)

        # calculate ones01
        ones01 = await self._measure_ones(tbm_specs, sim_dir, sim_db, dut, eval_ampl, 1.0, '01')

        # calculate ones10
        ones10 = await self._measure_ones(tbm_specs, sim_dir, sim_db, dut, eval_ampl, -1.0, '10')

        return dict(
            ones01=ones01,
            ones10=ones10,
        )

    async def _measure_ones(self, tbm_specs: Dict[str, Any], sim_dir: Path, sim_db: SimulationDB,
                            dut: DesignInstance, eval_ampl: float, sign: float, suf: str) -> float:
        """Helper function to set up testbench managers for calculating number of 'ones' at
        comparator output for 'num_points' number of evaluation points

        Parameters
        ----------
        tbm_specs : Dict[str, Any]
            Testbench manager specs
        sim_dir : Path
            Simulation directory
        sim_db : SimulationDB
            Simulation database
        dut : DesignInstance
            device under test
        eval_ampl : float
            Small positive / negative input amplitude during evaluation phase of overdrive recovery
        sign : float
            1.0 if differential input transitions from large negative to small positive
            -1.0 if differential input transitions from large positive to small negative
        suf : str
            '01' if sign is 1.0
            '10' if sign is -1.0

        Returns
        -------
        percentage of 'ones' at comparator output
        """
        specs = self.specs
        num_points = specs['num_points']
        ov_wave = specs['ov_wave_params']
        cmode, ampl = ov_wave['cmode'], ov_wave['ampl']
        n_rst, n_eval = ov_wave['n_rst'], ov_wave['n_eval']

        tbm_specs['sup_values']['VSS_in'] = cmode - sign * ampl
        tbm_specs['sup_values']['VDD_in'] = eval_ampl

        tbm = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm_specs))
        tbm.sim_params['t_sim'] = f't_rst + {num_points}*{n_rst + n_eval}*tper + tper'
        sim_id = f'overdrive_noise{suf}'
        sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id, dut, tbm, {})
        ones = tbm.calc_ones(sim_results.data, n_rst, num_points)

        return ones
