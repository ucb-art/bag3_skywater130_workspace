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

import importlib
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import math, copy
import scipy.interpolate as interp
from scipy.optimize import brentq, curve_fit

from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5
from bag.math.interpolate import LinearInterpolator
from bag.util.search import FloatBinaryIterator
from bag.util.immutable import Param


class CompACTB(TestbenchManager):
    """This class sets up the comparator AC measurement testbench using replica biasing.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_params: Param = self.specs['sim_params']
        sim_options: Param = self.specs['sim_options']
        freq_start: float = self.specs['freq_start']
        freq_stop: float = self.specs['freq_stop']
        num_per_dec: int = self.specs['num_per_dec']

        freq_num = math.ceil(math.log10(freq_stop) - math.log10(freq_start)) * num_per_dec

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='AC',
                           sweep=dict(type='LOG',
                                      start=freq_start,
                                      stop=freq_stop,
                                      num=freq_num,
                                      ),
                           options=dict(oppoint='screen'),
                           ),
                      dict(type='NOISE',
                           sweep=dict(type='LOG',
                                      start=freq_start,
                                      stop=freq_stop,
                                      num=freq_num,
                                      ),
                           in_probe='VINPUT',
                           p_port='XDUT_wrap.v_compM',
                           # p_port='v_OUTP',
                           n_port='0',
                           ),
                      ],
            params=sim_params,
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    def post_process_ac_noise(self, sim_data: SimData, plot_flag: bool = True) -> Dict[str, float]:
        # AC data
        sim_data.open_group('ac')
        freq_AC = sim_data['freq']
        v_compM = sim_data['XDUT_wrap.v_compM'][0]
        v_compP = sim_data['XDUT_wrap.v_compP'][0]
        v_sampP = sim_data['XDUT_wrap.v_SAMPP'][0]
        # v_SWN_P = sim_data['XDUT_wrap.v_SWN_P'][0]
        v_ACP = sim_data['XDUT_wrap.v_ACP'][0]
        v_ACM = sim_data['XDUT_wrap.v_ACM'][0]
        # v_in = sim_data['v_INP'][0]
        # v_out = sim_data['v_OUTP'][0]

        # v_compM_rep = sim_data['XDUT_wrap.XDUT.v_COMPM_rep'][0]
        # v_compP_rep = sim_data['XDUT_wrap.XDUT.v_COMPP_rep'][0]
        # v_ACP_rep = sim_data['XDUT_wrap.XDUT.v_ACP_rep'][0]
        # v_ACM_rep = sim_data['XDUT_wrap.XDUT.v_ACM_rep'][0]

        if plot_flag:
            plt.subplot(3, 1, 1)
            plt.semilogx(freq_AC, np.abs(v_compM/v_sampP), label='gain')
            # plt.semilogx(freq_AC, np.abs(v_out/v_in), label='gain')
            plt.legend()
            plt.xlabel('Frequency (in Hz)')
            plt.ylabel('Gain (V/V)')

            plt.subplot(3, 1, 2)
            plt.semilogx(freq_AC, np.abs(v_compM), label='compM')
            plt.semilogx(freq_AC, np.abs(v_compP), label='compP')
            plt.semilogx(freq_AC, np.abs(v_sampP), label='sampP')
            plt.semilogx(freq_AC, np.abs(v_ACP), label='ACP')
            plt.semilogx(freq_AC, np.abs(v_ACM), label='ACM')
            # plt.semilogx(freq_AC, np.abs(v_SWN_P), label='SWN_P')
            # plt.semilogx(freq_AC, np.abs(v_out), label='out')
            # plt.semilogx(freq_AC, np.abs(v_in), label='in')
            plt.legend()
            plt.xlabel('Frequency (in Hz)')
            plt.ylabel('Voltage (in V)')

            plt.subplot(3, 1, 3)
            plt.semilogx(freq_AC, np.angle(v_compM, deg=True), label='compM')
            plt.semilogx(freq_AC, np.angle(v_compP, deg=True), label='compP')
            plt.semilogx(freq_AC, np.angle(v_sampP, deg=True), label='sampP')
            plt.semilogx(freq_AC, np.angle(v_ACP, deg=True), label='ACP')
            plt.semilogx(freq_AC, np.angle(v_ACM, deg=True), label='ACM')
            # plt.semilogx(freq_AC, np.angle(v_out, deg=True), label='out')
            # plt.semilogx(freq_AC, np.angle(v_in, deg=True), label='in')
            plt.legend()
            plt.xlabel('Frequency (in Hz)')
            plt.ylabel('Angle (in deg)')

            # plt.subplot(4, 1, 4)
            # plt.semilogx(freq_AC, np.abs(v_compM_rep), label='compM')
            # plt.semilogx(freq_AC, np.abs(v_compP_rep), label='compP')
            # plt.semilogx(freq_AC, np.abs(v_ACP_rep), label='ACP')
            # plt.semilogx(freq_AC, np.abs(v_ACM_rep), label='ACM')
            # plt.legend()
            # plt.xlabel('Frequency (in Hz)')
            # plt.ylabel('Replica Voltage (in V)')

            plt.show()

        # Noise data
        sim_data.open_group('noise')
        freq = sim_data['freq']
        freq_log = np.log(freq)

        noise_out = sim_data['out'][0]
        noise_out_fun = LinearInterpolator([freq_log], np.log(noise_out ** 2), [1e-9],
                                           extrapolate=True)
        noise_out_V = np.sqrt(noise_out_fun.integrate(freq_log[0], freq_log[-1],
                                                      logx=True, logy=True, raw=True))

        noise_in = sim_data['in'][0]
        noise_in_fun = LinearInterpolator([freq_log], np.log(noise_in ** 2), [1e-9],
                                          extrapolate=True)
        noise_in_V = np.sqrt(noise_in_fun.integrate(freq_log[0], freq_log[-1],
                                                    logx=True, logy=True, raw=True))

        integ_noise_dict = dict(noise_out=noise_out_V,
                                noise_in=noise_in_V,)
        for sig in sim_data.signals:
            if not (sig.endswith('total') or sig == 'out' or sig == 'in' or sig == 'gain' or
                    0.0 in sim_data[sig][0]):
                noise_fun = LinearInterpolator([freq_log], np.log(sim_data[sig][0]), [1e-9],
                                               extrapolate=True)
                integ_noise_dict[sig] = np.sqrt(noise_fun.integrate(freq_log[0],
                                                freq_log[-1], logx=True, logy=True, raw=True))

        return integ_noise_dict


class CompTranTB(TestbenchManager):
    """This class sets up the comparator AC measurement testbench using replica biasing.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_params: Param = self.specs['sim_params']
        stop_time: float = self.specs['stop_time']

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='TRAN',
                           start=0.0,
                           stop=stop_time,
                           options=dict(errpreset='conservative'),
                           )
                      ],
            params=sim_params,
        )

        return netlist_info_from_dict(sim_setup)

    def post_process_tran(self, sim_data: SimData, plot_flag: bool = True) -> None:
        # TRAN data
        sim_data.open_group('tran')
        time = sim_data['time']
        v_compM = sim_data['XDUT_wrap.v_compM'][0]
        v_compP = sim_data['XDUT_wrap.v_compP'][0]
        v_sampP = sim_data['XDUT_wrap.v_SAMPP'][0]
        v_ACP = sim_data['XDUT_wrap.v_ACP'][0]
        v_ACM = sim_data['XDUT_wrap.v_ACM'][0]
        # v_in = sim_data['v_INP'][0]
        # v_out = sim_data['v_OUTP'][0]

        v_compM_rep = sim_data['XDUT_wrap.XDUT.v_COMPM_rep'][0]
        v_compP_rep = sim_data['XDUT_wrap.XDUT.v_COMPP_rep'][0]
        v_ACP_rep = sim_data['XDUT_wrap.XDUT.v_ACP_rep'][0]
        v_ACM_rep = sim_data['XDUT_wrap.XDUT.v_ACM_rep'][0]
        # print(f'Gain is {max(v_out - np.mean(v_out)) / max(v_in)}')

        if plot_flag:
            plt.plot(time, v_compM, label='compM')
            plt.plot(time, v_compP, label='compP')
            # plt.plot(time, v_sampP, label='sampP')
            plt.plot(time, v_ACP, label='ACP')
            plt.plot(time, v_ACM, label='ACM')
            # plt.plot(time, v_out - np.mean(v_out), label='out')
            # plt.plot(time, v_in, label='in')
            plt.legend()
            plt.xlabel('Time (in sec)')
            plt.ylabel('Voltage (in V)')

            plt.show()


class CompACMM(MeasurementManager):
    """This class measures AC parameters of comparator in amplification phase.

    This measurement is performed as follows:

    1. ...
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'ac'

    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:

        if state == 'ac':
            done = False
            next_state = 'tran'
            # output = {}
            noise_dict = cast(CompACTB, tb_manager).post_process_ac_noise(data)
            output = copy.deepcopy(noise_dict)

            noise_out = noise_dict['noise_out']
            del noise_dict['noise_out']
            noise_out_sq = noise_out ** 2
            print(f'Output noise is {noise_out} V')
            print(f'Input noise is {noise_dict["noise_in"]} V')
            del noise_dict['noise_in']

            sorted_components = sorted(noise_dict.items(), key=lambda kv: kv[1], reverse=True)
            print(f'Top noise components are:')
            for i in sorted_components[:5]:
                print(f'{i[0]} : {i[1] ** 2 / noise_out_sq * 100}%')
        elif state == 'tran':
            done = True
            next_state = ''
            output = {}

            cast(CompTranTB, tb_manager).post_process_tran(data)
        else:
            raise ValueError(f'Unknown state: {state}')

        return done, next_state, output
