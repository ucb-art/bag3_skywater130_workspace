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

from typing import Any, Sequence, Tuple, Mapping, Dict, Optional, Union

from pathlib import Path

from bag.io.file import write_yaml

from bag.concurrent.util import GatherHelper
from bag.simulation.core import TestbenchManager
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM


class DelayCharMM(MeasurementManager):
    """Characterize delay vs corners.

    Notes
    -----
    specification dictionary has the following entries:

    in_pin : str
        the input pin.
    out_pin : str
        the output pin.
    out_invert : bool
        Defaults to False.  True if the output is inverted.
    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            t_rf :
                input rise/fall time.
            c_load :
                load capacitance.
    sim_envs_swp_info : Mapping[str, Any]
        corner sweep information list.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._td_specs: Mapping[str, Any] = {}
        self._sup_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        out_invert: bool = specs['out_invert']
        tbm_specs: Mapping[str, Any] = specs['tbm_specs']

        self._sup_list = sorted(tbm_specs['sup_values'].keys())
        self._td_specs = dict(
            in_pin=in_pin,
            out_pin=out_pin,
            out_invert=out_invert,
            tbm_specs=tbm_specs,
            out_rise=True,
            out_fall=True,
        )

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        out_pin: str = specs['out_pin']
        sim_envs_swp_info: Mapping[str, Any] = specs['sim_envs_swp_info']

        name_format: str = sim_envs_swp_info['name_format']
        voltage_precision: int = sim_envs_swp_info['voltage_precision']
        voltage_types: Mapping[str, str] = sim_envs_swp_info['voltage_types']
        sim_envs: Sequence[Mapping[str, Any]] = sim_envs_swp_info['sim_envs']

        # make distinct measurement managers
        mm_table = {}
        result_table = {}
        voltage_fmt = '{:.%df}' % voltage_precision
        result_name_list = []
        for env_config in sim_envs:
            sim_env: str = env_config['sim_env']
            voltages: Mapping[str, float] = env_config['voltages']

            # get result name
            vstr_table = {k: voltage_fmt.format(v).replace('.', 'p') for k, v in voltages.items()}
            result_name = name_format.format(sim_env=sim_env, **vstr_table)
            self._make_td_mm(result_name, sim_db, dut, sim_env, voltage_fmt, voltage_types,
                             voltages, mm_table, result_table)

            result_name_list.append(result_name)

        # simulate in batch
        gatherer = GatherHelper()
        idx_map = {}
        for idx, (work_dir, mm) in enumerate(mm_table.items()):
            gatherer.append(sim_db.async_simulate_mm_obj(work_dir, sim_dir / work_dir, dut, mm))
            idx_map[work_dir] = idx

        results = await gatherer.gather_err()

        # post-process data
        td_table = {}
        rf_table = {}
        ans = {'td': td_table, 'trf': rf_table}
        for result_name, work_dir in result_table.items():
            idx = idx_map[work_dir]
            data = results[idx].data['timing_data'][out_pin]
            td_rise = data['cell_rise'].item()
            td_fall = data['cell_fall'].item()
            tr = data['rise_transition'].item()
            tf = data['fall_transition'].item()
            if td_rise < td_fall:
                td_table[result_name] = td_rise
                rf_table[result_name] = tr
            else:
                td_table[result_name] = td_fall
                rf_table[result_name] = tf

        write_yaml(sim_dir / f'{name}.yaml', ans)
        return ans

    def _make_td_mm(self, result_name: str, sim_db: SimulationDB,
                    dut: Optional[DesignInstance], sim_env: str, voltage_fmt: str,
                    voltage_types: Mapping[str, str], voltages: Mapping[str, float],
                    mm_table: Dict[str, MeasurementManager], result_table: Dict[str, str]) -> None:
        # get working directory
        str_list = [f'{k}_{voltage_fmt.format(voltages[voltage_types[k]]).replace(".", "p")}'
                    for k in self._sup_list]
        work_dir = f'td_{dut.cache_name}_{sim_env}_{"_".join(str_list)}'
        if work_dir not in mm_table:
            mm_specs = dict(**self._td_specs)
            mm_specs['tbm_specs'] = tbm_specs = dict(**mm_specs['tbm_specs'])
            tbm_specs['sup_values'] = sup_values = dict(**tbm_specs['sup_values'])

            # update measurement specs and create Measurement Manager
            tbm_specs['sim_envs'] = [sim_env]
            for sup_pin in list(sup_values.keys()):
                sup_values[sup_pin] = voltages[voltage_types[sup_pin]]
            mm = sim_db.make_mm(CombLogicTimingMM, mm_specs)

            mm_table[work_dir] = mm
        result_table[result_name] = work_dir

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')
