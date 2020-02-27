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

"""This package contains measurement class for comparators."""

from typing import Optional, Tuple, Dict, Any, Sequence, Union, List, cast

import math
import importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
from scipy.stats import norm
import matplotlib.pyplot as plt

from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData, AnalysisData, AnalysisType
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5
from bag.math.interpolate import LinearInterpolator
from bag.util.search import FloatBinaryIterator

from .tran import CompTranTB
from .pss import CompPSSTB


class InvAZCharMM(MeasurementManager):
    """This class measures inverter connected as auto zeroing amplifier based on transient, PSS,
    PAC, PNoise"""

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)

        # output parameters
        self.tau_rst: float = 0.0
        self.tau_regen: float = 0.0

        # setup for P
        self.output_PAC = None
        self.output_PNoise = None
        self.PNoise_run = False
        self.noise_time_point = None

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'tran'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tb_type = state
        tb_name = self.get_testbench_name(tb_type)
        tb_specs = self.get_testbench_specs(tb_type).copy()
        tb_params = self.get_default_tb_sch_params(tb_type)

        if state == 'tran':
            pass
        elif state == 'PAC':
            tb_specs['tb_type'] = 'PSS_PAC'
        elif state == 'PNoise':
            if self.PNoise_run:
                tb_specs['tb_type'] = 'PNoise'
                tb_specs['pnoise_params']['options']['noisetimepoints'] = [self.noise_time_point]
            else:
                tb_specs['tb_type'] = 'PSS'
        elif state == 'impulse':
            tb_specs['pwl_params']['tsim'] = self.output_PAC['time'][-1]
            tb_specs['pwl_params']['impulse_delay'] = self.output_PAC['impulse_delay']

        else:
            raise ValueError(f'Unknown state: {state}')

        return tb_name, tb_type, tb_specs, tb_params

    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:

        if state == 'tran':
            done = False
            next_state = 'PAC'
            output = {}

            # plot waveforms
            # self.process_tran_waveforms(data, plot_flag=True)

        elif state == 'PAC':
            done = False
            debug = False
            next_state = 'impulse' if debug else 'PNoise'
            output = {}

            # post process PAC waveforms
            self.output_PAC = cast(CompPSSTB, tb_manager).post_process_PAC(data, debug)
        elif state == 'impulse':
            done = False
            next_state = 'PNoise'
            output = {}

            # PAC signals
            calc = self.output_PAC['impulse_response']
            calc2 = self.output_PAC['output_wave']
            in2 = self.output_PAC['input_wave']
            time_PAC = self.output_PAC['time']
            clk_PAC = self.output_PAC['clk']
            PAC_idx = self.output_PAC['PAC_idx']

            # tran signals
            time = data['time']
            out = LinearInterpolator([time], data['v_OUTP'][0], [1e-12])(time_PAC)
            clk = LinearInterpolator([time], data['clk'][0], [1e-12])(time_PAC)
            vin = LinearInterpolator([time], data['Vin'][0], [1e-12])(time_PAC)
            clk_diff = (np.append(np.diff(clk), clk[0] - clk[-1]) > 0).astype(int)
            tran_idx = np.argmin(abs(clk_diff * clk - 0.5))

            roll = PAC_idx - tran_idx
            clk_roll = np.roll(clk, roll)
            out_roll = np.roll(out, roll)
            vin_roll = np.roll(vin, roll)

            # integ_calc = np.trapz(calc, time_PAC)
            # integ_real = np.trapz(out - out[0], time)

            plt.subplot(311)
            plt.plot(time_PAC, clk_roll, label='clk')
            plt.plot(time_PAC, clk_PAC, label='clk_PAC')
            plt.legend()

            plt.subplot(312)
            plt.plot(time_PAC, vin_roll - 0.15, label='input')
            plt.plot(time_PAC, in2, label='input_PAC')
            plt.legend()

            plt.subplot(313)
            # plt.plot(time_PAC, (out_roll - out_roll[0]) / 1.0e-14, label='real')
            plt.plot(time_PAC, out_roll, label='real')
            # plt.plot(time_PAC, calc, label='calc')
            plt.plot(time_PAC, calc2, label='calc')
            plt.legend()
            plt.show()

        elif state == 'PNoise':
            output = {}

            if self.PNoise_run:
                done = True
                next_state = ''

                # post process PNoise waveforms
                self.output_PNoise = cast(CompPSSTB, tb_manager).post_process_PNoise(data,
                                                                                    self.output_PAC)
            else:
                done = False
                next_state = 'PNoise'
                self.PNoise_run = True

                # post process PSS time domain waveforms
                data.open_group('pss_td')
                time = data['time']
                clk = data['clk'][0]
                clk_diff = (np.append(np.diff(clk), clk[0] - clk[-1]) > 0).astype(int)
                clk_idx = np.argmin(abs(clk_diff * clk - 0.5))
                rise_edge = time[clk_idx]
                self.noise_time_point = (rise_edge + self.output_PAC['time_max_gain']) % time[-1]
        else:
            raise ValueError(f'Unknown state: {state}')

        if done:
            noise_in = self.process_PSS_waveforms(self.output_PAC, self.output_PNoise,
                                                  plot_flag=True)
            output = dict(
                noise_in=noise_in,
            )
            print(output)

        return done, next_state, output

    @classmethod
    def process_tran_waveforms(cls, sim_data: SimData, plot_flag: bool = False) -> None:
        v_in = sim_data['Vin'][0]
        v_out = sim_data['v_OUTP'][0]
        v_samp = sim_data['XDUT_wrap.v_samp'][0]
        v_ac = sim_data['XDUT_wrap.v_ac'][0]
        clk = sim_data['clk'][0]

        time = sim_data['time']
        if plot_flag:
            plt.plot(time, clk, label='clk')
            plt.plot(time, v_in, label='v_in')
            plt.plot(time, v_out, label='v_out')
            plt.plot(time, v_samp, label='v_samp')
            plt.plot(time, v_ac, label='v_ac')
            plt.xlabel('time (in sec)')
            plt.ylabel('Voltage (in V)')
            plt.legend()
            plt.show()

    @classmethod
    def process_PSS_waveforms(cls, output_PAC: Dict[str, Any], output_PNoise: Dict[str, Any],
                              plot_flag: bool = False) -> float:
        # assume that PAC time is true time
        # PAC data
        time = output_PAC['time']

        clk_PAC = output_PAC['clk']
        # clkd = output_PAC['clkd']

        sig = output_PAC['sig']
        gain = output_PAC['gain']

        # PNoise data
        noise = output_PNoise['noise_out']

        # find max gain point
        gain_idx = np.argmax(np.abs(gain))
        out_noise_sq = noise ** 2
        # find input referred noise
        input_noise = np.abs(noise / gain[gain_idx])
        print(f'At max gain point, output referred noise = {noise} V')
        print(f'Max input referred noise = {input_noise} V')

        noise_components = output_PNoise['noise_components']
        sorted_components = sorted(noise_components.items(), key=lambda kv: kv[1], reverse=True)
        print(f'Top noise components are:')
        for i in sorted_components[:5]:
            print(f'{i[0]} : {i[1]**2 / out_noise_sq * 100}%')

        if plot_flag:
            plt.figure()
            plt.subplot(211)
            plt.plot(time, clk_PAC, label='clk_PAC')
            plt.plot(time, sig, label='compM')
            plt.xlabel('time (in sec)')
            plt.ylabel('Voltage (in V)')
            plt.legend()

            plt.subplot(212)
            plt.plot(time, gain, label='gain')
            plt.xlabel('time (in sec)')
            plt.ylabel('Absolute (V/V)')
            plt.legend()

            plt.show()

        return input_noise
