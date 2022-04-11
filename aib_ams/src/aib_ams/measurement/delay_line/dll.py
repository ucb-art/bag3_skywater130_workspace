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

from typing import Any, Sequence, Tuple, Set, Mapping, Dict, Iterable, Union, Optional

from pathlib import Path

import numpy as np

from pybag.core import get_cdba_name_bits

from bag.concurrent.util import GatherHelper

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_liberty.enum import TimingType

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.digital.flop.base import (
    FlopTimingBase, FlopInputMode, FlopMeasMode
)
from bag3_testbenches.measurement.digital.flop.array import FlopArrayTimingTB
from bag3_testbenches.measurement.digital.flop.char import FlopTimingCharMM


class FlopClkTimingTB(FlopTimingBase):
    """This class performs transient simulation to measure control signals timing constraints.

    Assumes maximum timing margin is t_clk_per/4.

    Notes
    -----
    specification dictionary has the following entries in addition to those in FlopTimingBase:

    flop_params : Mapping[str, Any]
        Flop parameters, with the following entries:

        in_pin : str
            the flop clk pin.
        out_pin : str
            the delay line output pin.
        clk_pin : str
            the delay line input pin.
        ctrl_pin : str
            the control pin(s).
        ctrl_code : int
            the nominal control setting.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._in_list: Sequence[str] = []

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        # NOTE: initialize in_list before running super's initialization,
        # so get_stimuli() behaves correctly.
        flop_params: Mapping[str, Any] = self.specs['flop_params']
        ctrl_pin: str = flop_params['ctrl_pin']

        self._in_list = get_cdba_name_bits(ctrl_pin)
        self._in_list.reverse()

        super().commit()

    @property
    def num_cycles(self) -> int:
        return 2

    @property
    def c_load_pins(self) -> Iterable[str]:
        return [self.flop_params['out_pin']]

    @classmethod
    def get_default_flop_params(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def get_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        return [FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=True, setup_rising=False,
                             hold_rising=False, meas_setup=True),
                FlopMeasMode(in_mode=FlopInputMode.IN, in_rising=False, setup_rising=False,
                             hold_rising=False, meas_setup=True),
                ]

    @classmethod
    def get_output_meas_modes(cls, flop_params: Mapping[str, Any]) -> Sequence[FlopMeasMode]:
        return []

    def get_stimuli(self) -> Tuple[Sequence[Mapping[str, Any]], Dict[str, int], Set[str],
                                   Sequence[str]]:
        mode = self.meas_mode
        flop_params = self.flop_params
        in_pin: str = flop_params['in_pin']
        out_pin: str = flop_params['out_pin']
        clk_pin: str = flop_params['clk_pin']
        ctrl_code: int = flop_params['ctrl_code']

        ctrl_pin = self._in_list[ctrl_code]

        clk_pulse = dict(pin=clk_pin, td='t_clk_delay-t_clk_per/4', tpw='t_clk_per/4', trf='t_rf',
                         tper='t_clk_per*5/4-t_setup')
        ctrl_pulse = dict(pin=ctrl_pin, tpw='t_sim', tper='2*t_sim', trf='t_rf',
                          td='t_clk_delay+t_clk_per*3/4-t_setup', pos=mode.input_rising)
        var_list = ['t_setup']
        pulses = [self.get_clk_pulse(in_pin, mode.is_pos_edge_clk), clk_pulse, ctrl_pulse]
        biases = {self._in_list[v]: 1 for v in range(0, ctrl_code)}
        for v in range(ctrl_code + 1, len(self._in_list)):
            biases[self._in_list[v]] = 0

        outputs = {ctrl_pin, in_pin, clk_pin, out_pin}
        return pulses, biases, outputs, var_list

    def get_output_map(self, output_timing: bool
                       ) -> Mapping[str, Tuple[Mapping[str, Any],
                                               Sequence[Tuple[EdgeType, Sequence[str]]]]]:
        if output_timing:
            return {}

        mode = self.meas_mode
        flop_params = self.flop_params
        in_pin: str = flop_params['in_pin']
        out_pin: str = flop_params['out_pin']
        clk_pin: str = flop_params['clk_pin']

        out_edge = EdgeType.FALL
        inc_delay = not mode.input_rising
        timing_info = self.get_timing_info(mode, [clk_pin], in_pin, '', True, inc_delay=inc_delay)
        edge_out_list = [(out_edge, [out_pin])]
        return {'t_setup': (timing_info, edge_out_list)}


class CtrlTimingMM(MeasurementManager):
    """Computes all timing constraints for DLL delayline

    Notes
    -----
    specification dictionary has the following entries:

    tbm_specs : Mapping[str, Any]
        DigitalTranTB related specifications.  The following simulation parameters are required:

            t_rst :
                reset duration.
            t_rst_rf :
                reset rise/fall time.
            t_bit :
                bit value duration.
            c_load :
                load capacitance.
    fake : bool
        Defaults to False.  True to return fake data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._clkin_specs: Mapping[str, Any] = {}
        self._flop_specs: Mapping[str, Any] = {}
        self._delay_shape: Tuple[int, int] = (0, 0)

        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        tbm_specs: Mapping[str, Any] = specs['tbm_specs']
        c_load: float = specs['c_load']
        t_clk_rf_flop: float = specs['t_clk_rf_flop']
        search_params: Mapping[str, Any] = specs['search_params']
        delay_thres: float = specs['delay_thres']
        delay_inc: float = specs['delay_inc']
        t_rf_list: Sequence[float] = specs['t_rf_list']
        t_clk_rf_list: Sequence[float] = specs['t_clk_rf_list']
        t_clk_rf_first: bool = specs['t_clk_rf_first']
        out_swp_info: Sequence[Any] = specs['out_swp_info']
        sim_env_name: str = specs['sim_env_name']
        constraint_min_map_clk: Mapping[Tuple[str, bool], float] = specs.get(
            'constraint_min_map_clk', None)
        constraint_min_map_flop: Mapping[Tuple[str, bool], float] = specs.get(
            'constraint_min_map_flop', None)
        fake_timing = specs.get('fake_timing', False)
        fake = specs.get('fake', False)
        
        fake_timing = fake or fake_timing

        new_tbm_specs = dict(**tbm_specs)
        new_tbm_specs['swp_info'] = []
        self._clkin_specs = dict(
            tbm_cls=FlopClkTimingTB,
            tbm_specs=new_tbm_specs,
            flop_params=dict(
                in_pin='dlyin',
                out_pin='dlyout',
                clk_pin='CLKIN',
                ctrl_pin='bk<63:0>',
                ctrl_code=0,
            ),
            delay_thres=float('inf'),
            delay_inc=delay_inc,
            constraint_min_map=constraint_min_map_clk,
            c_load=c_load,
            t_rf_list=[t_clk_rf_flop],
            t_clk_rf_list=t_clk_rf_list,
            t_clk_rf_first=t_clk_rf_first,
            out_swp_info=[],
            search_params=search_params,
            fake=fake_timing,
        )

        self._flop_specs = dict(
            tbm_cls=FlopArrayTimingTB,
            tbm_specs=new_tbm_specs,
            flop_params=dict(
                in_pin='bk<0:63>',
                out_pin='flop_q<0:62>,SOOUT',
                clk_pin='CLKIN',
                se_pin='iSE',
                si_pin='iSI',
                out_invert=False,
                clk_rising=True,
                out_timing_pin='SOOUT',
                c_load_pin='SOOUT',
            ),
            c_load=c_load,
            delay_thres=delay_thres,
            constraint_min_map=constraint_min_map_flop,
            sim_env_name=sim_env_name,
            t_rf_list=t_rf_list,
            t_clk_rf_list=[t_clk_rf_flop],
            t_clk_rf_first=t_clk_rf_first,
            out_swp_info=out_swp_info,
            search_params=search_params,
            fake=fake_timing,
        )

        if t_clk_rf_first:
            self._delay_shape = (len(t_clk_rf_list), len(t_rf_list))
        else:
            self._delay_shape = (len(t_rf_list), len(t_clk_rf_list))

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        specs = self.specs
        sim_env_name: str = specs['sim_env_name']
        t_clk_per: float = specs['t_clk_per']
        t_dist_err: float = specs['t_dist_err']
        t_max_pi: Mapping[str, float] = specs['t_max_pi']
        t_dist_targ: Mapping[str, float] = specs['t_dist_targ']
        t_setup_clkin_delta: float = specs['t_setup_clkin_delta']

        mm_clkin = sim_db.make_mm(FlopTimingCharMM, self._clkin_specs)
        mm_flop = sim_db.make_mm(FlopTimingCharMM, self._flop_specs)

        ctrl_id = f'seq_ctrl_{dut.cache_name}'
        gatherer = GatherHelper()
        sim_id = f'seq_clkin_{dut.cache_name}'
        gatherer.append(sim_db.async_simulate_mm_obj(sim_id, sim_dir / sim_id, dut, mm_clkin))
        gatherer.append(sim_db.async_simulate_mm_obj(ctrl_id, sim_dir / ctrl_id, dut, mm_flop))

        results = await gatherer.gather_err()

        data_clkin = results[0].data
        data_flop = results[1].data

        # compute clkin setup time as function of dlyin rise/fall time
        t_setup_dl_table = data_clkin['CLKIN'][0]['data']
        t_setup_dl = np.maximum(t_setup_dl_table['rise_constraint'],
                                t_setup_dl_table['fall_constraint'])
        t_setup_clkin = t_setup_dl - (t_clk_per / 4 - t_max_pi[sim_env_name] -
                                      t_dist_targ[sim_env_name]) / 2 + 2 * t_dist_err
        t_setup_clkin += t_setup_clkin_delta
        t_setup_clkin = np.broadcast_to(t_setup_clkin, self._delay_shape)

        # ctrl_related = 'dlyin'
        # setup_delta = 4 * t_dist_err - (t_clk_per / 4 - t_dist_targ[sim_env_name])
        # hold_delta = t_clk_per / 2 - t_setup_clkin + 2 * t_dist_err
        ctrl_related = 'CLKIN'
        setup_delta = hold_delta = 2 * t_dist_err
        ans = dict(
            CLKIN=[
                dict(related='dlyin',
                     timing_type=TimingType.setup_falling.name,
                     cond='RSTb',
                     data=dict(rise_constraint=t_setup_clkin),
                     ),
            ],
            SOOUT=data_flop['SOOUT'],
        )

        # compute flop input setup/hold time
        for idx in range(64):
            ctrl_pin = f'bk<{idx}>'
            self._record_flop_setup_hold(data_flop, ctrl_pin, 'RSTb && !iSE',
                                         setup_delta, hold_delta, ctrl_related, ans)
        self._record_flop_setup_hold(data_flop, 'iSI', 'RSTb && iSE', setup_delta, hold_delta,
                                     ctrl_related, ans)
        self._record_flop_setup_hold(data_flop, 'iSE', 'RSTb && iSE', setup_delta, hold_delta,
                                     ctrl_related, ans)

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

    def _record_flop_setup_hold(self, data_flop: Mapping[str, Any], pin_name: str, cond: str,
                                setup_delta: float, hold_delta: float, related: str,
                                ans: Dict[str, Any]) -> None:
        timing_list = data_flop[pin_name]
        table0 = timing_list[0]
        table1 = timing_list[1]
        if 'setup' in table0['timing_type']:
            data_setup = table0['data']
            data_hold = table1['data']
        else:
            data_setup = table1['data']
            data_hold = table0['data']

        setup_r = np.broadcast_to(data_setup['rise_constraint'], self._delay_shape)
        setup_f = np.broadcast_to(data_setup['fall_constraint'], self._delay_shape)
        hold_r = np.broadcast_to(data_hold['rise_constraint'], self._delay_shape)
        hold_f = np.broadcast_to(data_hold['fall_constraint'], self._delay_shape)
        # NOTE: broadcast_to returns read-only arrays
        setup_r = setup_r + setup_delta
        setup_f = setup_f + setup_delta
        hold_r = hold_r + hold_delta
        hold_f = hold_f + hold_delta

        ans[pin_name] = [dict(related=related, timing_type=TimingType.setup_rising.name,
                              cond=cond, data=dict(rise_constraint=setup_r,
                                                   fall_constraint=setup_f)),
                         dict(related=related, timing_type=TimingType.hold_rising.name,
                              cond=cond, data=dict(rise_constraint=hold_r,
                                                   fall_constraint=hold_f))]
