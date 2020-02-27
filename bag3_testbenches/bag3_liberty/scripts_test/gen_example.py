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

from typing import Dict, Any, Optional

import numpy as np

from bag3_liberty.data import Library
from bag3_liberty.enum import LogicType


def run_main():
    output_file = 'test.lib'

    lib_config = dict(
        name='test_library',
        revision='2019.0',
        precision=4,
        units=dict(
            voltage=1.0,
            current=1e-3,
            time=1e-12,
            capacitance=1e-15,
            resistance=1e3,
            power=1e-6,
        ),
        voltages=dict(
            VDD=1.0,
            VSS=0.0,
        ),
        sim_envs=[
            dict(name='tt_25_v1d0', process=1.0, temperature=25, voltage=1.0),
        ],
        lut=dict(
            CONSTRAINT=[
                [5e-12, 100e-12, 1000e-12],
                [5e-12, 20e-12, 100e-12, 1000e-12],
            ],
            DELAY=[
                [5e-12, 100e-12, 1000e-12],
                [1e-15, 100e-15, 1000e-15, 10000e-15],
            ],
        ),
        norm_drv_wvfm=dict(
            ndw0=dict(
                name='driver_waveform_default_rise',
                idx1=[5e-12, 10e-12, 15e-12],  # input transition time
                idx2=[0, 0.2, 0.4, 0.6, 0.8, 1.0],  # voltages normalized to VDD
                val=np.array([[0, 1e-12, 2e-12, 3e-12, 4e-12, 5e-12],
                              [0, 2e-12, 4e-12, 6e-12, 8e-12, 10e-12],
                              [0, 4e-12, 8e-12, 12e-12, 16e-12, 20e-12]]),
                # time when voltage reaches idx2 values
            ),
            ndw1=dict(
                name='driver_waveform_default_fall',
                idx1=[5e-12, 10e-12, 15e-12],  # input transition time
                idx2=[0, 0.25, 0.5, 0.75, 1.0],  # voltages normalized to VDD
                val=np.array([[0, 1e-12, 2e-12, 3e-12, 4e-12],
                              [0, 2e-12, 4e-12, 6e-12, 8e-12],
                              [0, 4e-12, 8e-12, 12e-12, 16e-12]]),
                # time when voltage reaches idx2 values
            )
        ),
        thresholds=dict(
            slew_derate=1,
            input_fall=50,
            input_rise=50,
            output_fall=50,
            output_rise=50,
            lower_fall=20,
            lower_rise=20,
            upper_fall=80,
            upper_rise=80,
        ),
        defaults=dict(
            threshold_voltage_group='nom',
            fanout_load=1,
            cell_leakage_power=0,
            inout_pin_cap=0,
            input_pin_cap=0,
            output_pin_cap=0,
            leakage_power_density=0,
            max_transition=5.0e-9,
        ),
    )

    cell_config = dict(
        name='TestCell',
        pwr_pins=dict(
            VDD='VDD',
        ),
        gnd_pins=dict(
            VSS='VSS',
        ),
    )

    input_pins = [
        dict(
            name='clock',
            logic='seq',
            cap_dict=dict(
                cap=10e-15,
                cap_fall=9e-15,
                cap_fall_range=(8e-15, 10e-15),
                cap_rise=11e-15,
                cap_rise_range=(10e-15, 12e-15),
            ),
            is_clock=True,
            max_trf=200e-12,
            pwr='VDD',
            gnd='VSS',
            dw_rise='driver_waveform_default_rise',
            dw_fall='driver_waveform_default_fall',
        ),
        dict(
            name='foo',
            logic='seq',
            cap_dict=dict(
                cap=10e-15,
                cap_fall=9e-15,
                cap_fall_range=(8e-15, 10e-15),
                cap_rise=11e-15,
                cap_rise_range=(10e-15, 12e-15),
            ),
            is_clock=False,
            max_trf=200e-12,
            pwr='VDD',
            gnd='VSS',
            timing=dict(
                clk='clock',
                cond='x and not y and not z',
                data=dict(
                    setup_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    setup_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    hold_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                        [25e-12, 30e-12, 50e-12, 120e-12],
                                        [40e-12, 50e-12, 80e-12, 200e-12]]),
                    hold_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                        [25e-12, 30e-12, 50e-12, 120e-12],
                                        [40e-12, 50e-12, 80e-12, 200e-12]]),
                ),
            ),
        ),
        dict(
            name='bus<9:0>',
            logic='seq',
            cap_dict=dict(
                cap=10e-15,
                cap_fall=9e-15,
                cap_fall_range=(8e-15, 10e-15),
                cap_rise=11e-15,
                cap_rise_range=(10e-15, 12e-15),
            ),
            is_clock=False,
            max_trf=200e-12,
            pwr='VDD',
            gnd='VSS',
            timing=dict(
                clk='clock',
                cond='x and y and z',
                data=dict(
                    setup_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    setup_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    hold_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                        [25e-12, 30e-12, 50e-12, 120e-12],
                                        [40e-12, 50e-12, 80e-12, 200e-12]]),
                    hold_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                        [25e-12, 30e-12, 50e-12, 120e-12],
                                        [40e-12, 50e-12, 80e-12, 200e-12]]),
                ),
            ),
        ),
    ]

    output_pins = [
        dict(
            name='bar',
            logic='seq',
            cap_dict=dict(
                cap_max=100e-15,
                cap_min=1e-15,
            ),
            pwr='VDD',
            gnd='VSS',
            func='IQ',
        ),
        dict(
            name='fout',
            logic='seq',
            cap_dict=dict(
                cap_max=100e-15,
                cap_min=1e-15,
            ),
            pwr='VDD',
            gnd='VSS',
            func='IQ',
            timing=dict(
                clk='clock',
                cond='x and not y and z',
                data=dict(
                    delay_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    trf_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                       [25e-12, 30e-12, 50e-12, 120e-12],
                                       [40e-12, 50e-12, 80e-12, 200e-12]]),
                    delay_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    trf_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                       [25e-12, 30e-12, 50e-12, 120e-12],
                                       [40e-12, 50e-12, 80e-12, 200e-12]]),
                )
            )
        ),
        dict(
            name='fout_comb0',
            logic='comb',
            cap_dict=dict(
                cap_max=100e-15,
                cap_min=1e-15,
            ),
            pwr='VDD',
            gnd='VSS',
            func='!((x & y) | z)',
            timing=dict(
                clk='clock',
                cond='not x and y and not z',
                data=dict(
                    delay_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    trf_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                       [25e-12, 30e-12, 50e-12, 120e-12],
                                       [40e-12, 50e-12, 80e-12, 200e-12]]),
                )
            )
        ),
        dict(
            name='fout_comb1',
            logic='comb',
            cap_dict=dict(
                cap_max=100e-15,
                cap_min=1e-15,
            ),
            pwr='VDD',
            gnd='VSS',
            func='(x | y) & !z',
            timing=dict(
                clk='clock',
                sense='neg',
                cond='not x and y and z',
                data=dict(
                    delay_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    trf_rise=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                       [25e-12, 30e-12, 50e-12, 120e-12],
                                       [40e-12, 50e-12, 80e-12, 200e-12]]),
                    delay_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                         [25e-12, 30e-12, 50e-12, 120e-12],
                                         [40e-12, 50e-12, 80e-12, 200e-12]]),
                    trf_fall=np.array([[20e-12, 25e-12, 40e-12, 100e-12],
                                       [25e-12, 30e-12, 50e-12, 120e-12],
                                       [40e-12, 50e-12, 80e-12, 200e-12]]),
                )
            )
        )
    ]

    lib = Library(lib_config)
    cell = lib.create_cell(**cell_config)

    for info in input_pins:
        pin = cell.create_input_pin(info['name'], info['logic'], info['cap_dict'], info['is_clock'],
                                    info['max_trf'], info['pwr'], info['gnd'],
                                    info.get('dw_rise', ''), info.get('dw_fall', ''))
        timing: Optional[Dict[str, Any]] = info.get('timing', None)
        if timing is not None:
            pin.add_timing(timing['clk'], timing['data'], timing.get('sense', 'non_unate'),
                           timing['cond'])

    for info in output_pins:
        pin = cell.create_output_pin(info['name'], info['logic'], info['cap_dict'], info['pwr'],
                                     info['gnd'], info.get('func', ''))
        timing = info.get('timing', None)
        if timing is not None:
            pin.add_timing(timing['clk'], timing['data'], timing.get('sense', 'non_unate'),
                           timing['cond'])

    lib.generate(output_file)


if __name__ == '__main__':
    run_main()
