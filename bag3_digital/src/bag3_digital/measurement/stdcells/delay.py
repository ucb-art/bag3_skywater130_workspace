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

from typing import Dict, Any, Tuple, Optional, Union, Mapping

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import DesignInstance, SimulationDB, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_testbenches.measurement.digital.comb import CombLogicTimingMM
from bag3_testbenches.measurement.digital.delay import RCDelayCharMM

from ..cap.delay_match import CapDelayMatch
from ..util import get_in_buffer_pin_names


class BufferRCDelayCharMM(MeasurementManager):
    """Characterize delay of a digital gate with input buffers.

    Notes
    -----
    specification dictionary has the following entries entries:

    in_pin : str
        input pin.
    out_pin : str
        output pin.
    out_invert : Union[bool, Sequence[bool]]
        True if output is inverted from input.  Corresponds to each input/output pair.
    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            t_rf :
                input rise/fall time
            c_load :
                load capacitance.
    buf_config : Mapping[str, Any]
        input buffer configuration parameters.
    search_params : Mapping[str, Any]
        input capacitance search parameters.
    c_load : float
        nominal load capacitance.
    scale_min : float
        lower bound scale factor for c_load.
    scale_max : float
        upper bound scale factor for c_load.
    num_samples : int
        number of data points to measure.
    wait_cycles : int
        Defaults to 0.  Number of cycles to wait toggle before finally measuring delay.
    t_unit : float
        Defaults to 1e-12 (1 ps).  The estimated time unit.  Used to normalize matrices.
    plot : bool
        Defaults to False.  True to plot fitted lines.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cin_mm: Optional[CapDelayMatch] = None
        self._td_specs: Dict[str, Any] = {}

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        specs = self.specs
        in_pin: str = specs['in_pin']
        out_pin: str = specs['out_pin']
        out_invert: bool = specs['out_invert']
        tbm_specs: Mapping[str, Any] = specs['tbm_specs']
        buf_config: Mapping[str, Any] = specs['buf_config']
        search_params: Mapping[str, Any] = specs['search_params']
        c_load: float = specs['c_load']
        scale_min: float = specs['scale_min']
        scale_max: float = specs['scale_max']
        num_samples: int = specs['num_samples']
        wait_cycles: int = specs.get('wait_cycles', 0)

        cin_specs = dict(
            in_pin=in_pin,
            buf_config=buf_config,
            search_params=search_params,
            tbm_specs=tbm_specs,
            load_list=[dict(pin=out_pin, type='cap', value='c_load')],
        )

        self._cin_mm = self.make_mm(CapDelayMatch, cin_specs)
        self._cin_mm.specs['tbm_specs']['sim_params']['c_load'] = c_load

        td_tbm_specs = dict(**tbm_specs)
        td_tbm_specs['swp_info'] = [('c_load', dict(type='LOG', start=c_load * scale_min,
                                                    stop=c_load * scale_max, num=num_samples))]
        self._td_specs = dict(
            in_pin=in_pin,
            start_pin=get_in_buffer_pin_names(in_pin)[1],
            out_pin=out_pin,
            out_invert=out_invert,
            out_rise=True,
            out_fall=True,
            wait_cycles=wait_cycles,
            tbm_specs=td_tbm_specs,
        )

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        out_pin: str = specs['out_pin']
        t_unit: float = specs.get('t_unit', 1.0e-12)
        plot: bool = specs.get('plot', False)

        cin_result = await sim_db.async_simulate_mm_obj(f'{name}_cin', sim_dir / 'cin', dut,
                                                        self._cin_mm)
        cin_data = cin_result.data
        cin_rise: float = cin_data['cap_rise']
        cin_fall: float = cin_data['cap_fall']

        # get output res/cap
        self._td_specs['wrapper_params'] = self._cin_mm.wrapper_params

        delay_mm = self.make_mm(CombLogicTimingMM, self._td_specs)
        mm_output = await sim_db.async_simulate_mm_obj(f'{name}_td', sim_dir / 'td', dut, delay_mm)
        mm_result = mm_output.data

        sim_envs = mm_result['sim_envs']
        sim_params = mm_result['sim_params']
        c_load = sim_params['c_load']

        delay_data = mm_result['timing_data'][out_pin]
        td_rise: np.ndarray = delay_data['cell_rise']
        td_fall: np.ndarray = delay_data['cell_fall']

        r_fall, c_fall = RCDelayCharMM.fit_rc_out(td_fall, c_load, t_unit)
        r_rise, c_rise = RCDelayCharMM.fit_rc_out(td_rise, c_load, t_unit)
        if plot:
            from matplotlib import pyplot as plt

            plt.figure(1)
            plt.title(sim_envs[0])
            cl = c_load[0, ...]
            plt.plot(cl, td_rise[0, ...], 'bo', label='td_rise')
            plt.plot(cl, td_fall[0, ...], 'ko', label='td_fall')
            plt.plot(cl, r_rise.item() * (cl + c_rise.item()), '-r', label='td_rise_fit')
            plt.plot(cl, r_fall.item() * (cl + c_fall.item()), '-g', label='td_fall_fit')
            plt.legend()
            plt.show()

        ans = dict(
            sim_env=sim_envs[0],
            c_in=(cin_fall, cin_rise),
            r_out=(r_fall.item(), r_rise.item()),
            c_out=(c_fall.item(), c_rise.item()),
        )
        return ans

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')


def _fit_rc(td: np.ndarray, cl: np.ndarray, t_unit: float) -> Tuple[float, float]:
    r0, t0, _, _, _ = linregress(cl / t_unit, td / t_unit)
    c0 = t0 * t_unit / r0
    return r0, c0
