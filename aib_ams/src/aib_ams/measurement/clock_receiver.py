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

from typing import Any, Union, Tuple, Optional, Mapping, Dict, cast

import pprint

import numpy as np

from bag.simulation.core import TestbenchManager
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo

from bag3_liberty.enum import TimingType

from bag3_testbenches.measurement.data.tran import EdgeType
from bag3_testbenches.measurement.tran.digital import DigitalTranTB
from bag3_testbenches.measurement.digital.util import setup_digital_tran


class ClockReceiverMM(MeasurementManager):
    """Measures measure/launch delay.

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
        super().__init__(*args, **kwargs)
        self._tbm_info: Optional[Tuple[DigitalTranTB, Mapping[str, Any]]] = None

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        specs = self.specs
        fake: bool = specs.get('fake', False)

        load_list = [dict(pin='v_out', type='cap', value='c_load')]
        sine_list = [dict(pin='v_inp', amp='amp', freq='clk_freq', dc='in_dc')]
        diff_list = [(['v_inp'], ['v_inn'])]
        pin_values = dict()
        save_outputs = ['v_inp', 'v_inn', 'v_out']
        tbm_specs, tb_params = setup_digital_tran(specs, dut, sine_list=sine_list,
                                                  load_list=load_list, pin_values=pin_values,
                                                  save_outputs=save_outputs, diff_list=diff_list)
        tbm = cast(DigitalTranTB, sim_db.make_tbm(DigitalTranTB, tbm_specs))

        if fake:
            result = dict()
            return True, MeasInfo('done', result)

        tbm.sim_params['t_sim'] = 3 / float(tbm.sim_params['clk_freq'])
        self._tbm_info = tbm, tb_params
        return False, MeasInfo('sim', {})

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        return self._tbm_info, True

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        tbm: DigitalTranTB = self._tbm_info[0]

        sim_params = tbm.sim_params
        t_bit = 1/sim_params['clk_freq']
        t_start = t_bit * 1.5  # Start the measurement at the 1.5x tbit
        sin_thresh = sim_params['in_dc']

        data = cast(SimResults, sim_results).data

        tinr = tbm.calc_cross(data, 'v_inp', EdgeType.RISE, t_start=t_start, abs_thresh=sin_thresh)
        toutr = tbm.calc_cross(data, 'v_out', EdgeType.RISE, t_start=tinr)
        toutrf = tbm.calc_trf(data, 'v_out', True, t_start=tinr)
        tinf = tbm.calc_cross(data, 'v_inp', EdgeType.FALL, t_start=t_start, abs_thresh=sin_thresh)
        toutf = tbm.calc_cross(data, 'v_out', EdgeType.FALL, t_start=tinf)
        toutfr = tbm.calc_trf(data, 'v_out', False, t_start=tinf)

        result = dict(tdr=toutr-tinr, tdf=toutf-tinf, tr=toutrf, tf=toutfr)
        self.log(f'result:\n{pprint.pformat(result, width=100)}')

        return True, MeasInfo('done', result)

