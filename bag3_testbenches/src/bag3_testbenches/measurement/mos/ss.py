# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

"""This package contains measurement class for transistors."""

from typing import Optional, Tuple, Dict, Any, Sequence, Union, List, Mapping, Type

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData, AnalysisData, AnalysisType
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5
from bag.math.interpolate import LinearInterpolator

from ..data.util import brentq_safe


class MOSIdTB(TestbenchManager):
    """This class sets up the transistor drain current measurement testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'mos_tb_ibias')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vgs_max: float = specs['vgs_max']
        vgs_num: int = specs['vgs_num']
        is_nmos: bool = specs['is_nmos']
        vgs_min: float = specs.get('vgs_min', 0.0)
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        if is_nmos:
            start = vgs_min
            stop = vgs_max
        else:
            start = -vgs_max
            stop = -vgs_min

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='DC',
                           param='vgs',
                           sweep=dict(type='LINEAR',
                                      start=start,
                                      stop=stop,
                                      num=vgs_num,
                                      ),
                           save_outputs=['VD:p'],
                           ),
                      ],
            params=dict(vs=0.0),
            options=sim_options,
        )
        return netlist_info_from_dict(sim_setup)

    @classmethod
    def get_data(cls, data: SimData, specs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        ibias_min_fg: float = specs['ibias_min_fg']
        ibias_max_fg: float = specs['ibias_max_fg']
        vgs_res: float = specs['vgs_resolution']
        fg: int = specs['fg']
        is_nmos: bool = specs['is_nmos']

        # invert PMOS ibias sign
        ibias_sgn = -1.0 if is_nmos else 1.0

        data.open_analysis(AnalysisType.DC)
        vgs = data['vgs']
        ibias = data['VD:p'] * ibias_sgn

        fun_list = []
        vgs_max = vgs0 = vgs[0]
        vgs_min = vgs1 = vgs[-1]
        if is_nmos:
            ibias0 = ibias_min_fg * fg
            ibias1 = ibias_max_fg * fg
        else:
            ibias0 = ibias_max_fg * fg
            ibias1 = ibias_min_fg * fg
        for idx in range(ibias.shape[0]):
            interp_fun = InterpolatedUnivariateSpline(vgs, ibias[idx, :])
            fun_list.append(interp_fun)
            vgs_min = min(vgs_min, brentq_safe(interp_fun, vgs0, vgs1, ibias0))
            vgs_max = max(vgs_max, brentq_safe(interp_fun, vgs0, vgs1, ibias1))

        vgs_min = math.floor(vgs_min / vgs_res) * vgs_res
        vgs_max = math.ceil(vgs_max / vgs_res) * vgs_res
        num = int(math.ceil((vgs_max - vgs_min) / vgs_res)) + 1
        xvec = np.linspace(vgs_min, vgs_max, num, endpoint=True)
        ymat = np.stack([fun(xvec) for fun in fun_list], axis=0)
        return xvec, ymat

    def print_results(self, data: SimData) -> None:
        specs = self.specs
        ibias_min_fg: float = specs['ibias_min_fg']
        ibias_max_fg: float = specs['ibias_max_fg']

        xvec, imat = self.get_data(data, specs)

        print(f'for ibias_min_fg={ibias_min_fg:.4g}, ibias_max_fg={ibias_max_fg:.4g}, '
              f'vgs_min={xvec[0]:.4g}, vgs_max={xvec[-1]:.4g}')

        import matplotlib.pyplot as plt
        plt.figure()
        for idx, env in enumerate(data.sim_envs):
            plt.plot(xvec, imat[idx, :], label=env)

        plt.legend()
        plt.show()


class MOSSSTB(TestbenchManager):
    """This class sets up the transistor S parameter measurement testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'mos_tb_sp')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vbs_val: Union[float, List[float]] = specs['vbs']
        vgs_min: float = specs['vgs_min']
        vgs_max: float = specs['vgs_max']
        vgs_num: int = specs['vgs_num']
        vds_min: float = specs['vds_min']
        vds_max: float = specs['vds_max']
        vds_num: int = specs['vds_num']
        sp_freq: float = specs['sp_freq']
        is_nmos: bool = specs['is_nmos']
        fix_vbs_sign: bool = specs['fix_vbs_sign']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        vds_start, vds_stop, vb_dc, vbs_list = _fix_vbias_signs(vgs_min, vds_min, vds_max, vbs_val,
                                                                is_nmos, fix_vbs_sign)

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='DC',
                           param='vgs',
                           sweep=dict(type='LINEAR',
                                      start=vgs_min,
                                      stop=vgs_max,
                                      num=vgs_num,
                                      ),
                           ),
                      dict(type='SP',
                           param='vgs',
                           sweep=dict(type='LINEAR',
                                      start=vgs_min,
                                      stop=vgs_max,
                                      num=vgs_num,
                                      ),
                           freq=sp_freq,
                           ports=['PORTG', 'PORTD', 'PORTS'],
                           param_type='Y',
                           ),
                      ],
            params=dict(vb_dc=vb_dc),
            swp_info=[
                ['vbs', dict(type='LIST', values=vbs_list)],
                ['vds', dict(type='LINEAR', start=vds_start, stop=vds_stop, num=vds_num)]
            ],
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def get_ss_params(cls, data: SimData, specs: Dict[str, Any]) -> SimData:
        fg: int = specs['fg']
        sp_freq: float = specs['sp_freq']
        cfit_method: str = specs['cfit_method']
        is_nmos: bool = specs['is_nmos']

        ibias_sgn = -1.0 if is_nmos else 1.0

        # invert PMOS ibias sign
        data.open_analysis(AnalysisType.SP)
        ss_dict = cls.mos_y_to_ss(data, sp_freq, fg, cfit_method=cfit_method)
        data.open_analysis(AnalysisType.DC)
        ibias = data['VD:p'] * ibias_sgn
        ss_dict['ibias'] = ibias / fg
        # construct new SS parameter result dictionary
        ss_swp_names = data.sweep_params
        for key in ss_swp_names[1:]:
            ss_dict[key] = data[key]

        ana_data = AnalysisData(ss_swp_names, ss_dict, True)
        return SimData(data.sim_envs, {'ss': ana_data})

    @classmethod
    def mos_y_to_ss(cls, sim_data: SimData, char_freq: float, fg: int,
                    cfit_method: str = 'average') -> Dict[str, np.ndarray]:
        """Convert transistor Y parameters to small-signal parameters.

        This function computes MOSFET small signal parameters from 3-port
        Y parameter measurements done on gate, drain and source, with body
        bias fixed.  This functions fits the Y parameter to a capcitor-only
        small signal model using least-mean-square error.

        Parameters
        ----------
        sim_data : SimData
            simulation data.
        char_freq : float
            the frequency Y parameters are measured at.
        fg : int
            number of transistor fingers used for the Y parameter measurement.
        cfit_method : str
            method used to extract capacitance from Y parameters.  Currently
            supports 'average' or 'worst'

        Returns
        -------
        ss_dict : Dict[str, np.ndarray]
            A dictionary of small signal parameter values stored as numpy
            arrays.  These values are normalized to 1-finger transistor.
        """
        w = 2 * np.pi * char_freq

        y11 = sim_data['y11']
        y12 = sim_data['y12']
        y13 = sim_data['y13']
        y21 = sim_data['y21']
        y22 = sim_data['y22']
        y23 = sim_data['y23']
        y31 = sim_data['y31']
        y32 = sim_data['y32']
        y33 = sim_data['y33']

        gm = (y21.real - y31.real) / 2.0
        gds = (y22.real - y32.real) / 2.0
        gb = (y33.real - y23.real) / 2.0 - gm - gds

        cgd12 = -y12.imag / w
        cgd21 = -y21.imag / w
        cgs13 = -y13.imag / w
        cgs31 = -y31.imag / w
        cds23 = -y23.imag / w
        cds32 = -y32.imag / w
        cgg = y11.imag / w
        cdd = y22.imag / w
        css = y33.imag / w

        if cfit_method == 'average':
            cgd = (cgd12 + cgd21) / 2
            cgs = (cgs13 + cgs31) / 2
            cds = (cds23 + cds32) / 2
        elif cfit_method == 'worst':
            cgd = np.maximum(cgd12, cgd21)
            cgs = np.maximum(cgs13, cgs31)
            cds = np.maximum(cds23, cds32)
        else:
            raise ValueError('Unknown cfit_method = %s' % cfit_method)

        cgb = cgg - cgd - cgs
        cdb = cdd - cds - cgd
        csb = css - cgs - cds
        return dict(
            gm=gm / fg,
            gds=gds / fg,
            gb=gb / fg,
            cgd=cgd / fg,
            cgs=cgs / fg,
            cds=cds / fg,
            cgb=cgb / fg,
            cdb=cdb / fg,
            csb=csb / fg,
        )

    def print_results(self, data: SimData) -> None:
        specs = self.specs
        vgs_test: Optional[float] = specs.get('vgs_test', None)
        vds_test: Optional[float] = specs.get('vds_test', None)
        vbs_test: float = specs.get('vbs_test', 0.0)
        sim_env_test: str = specs.get('sim_env_test', data.sim_envs[0])

        eidx = data.sim_envs.index(sim_env_test)
        vgs_vec = data['vgs']
        vds_vec = data['vds']
        vbs_vec = data['vbs']

        if vgs_test is None:
            vgs_idx = vgs_vec.size // 2
        else:
            vgs_idx = np.argmin(np.abs(vgs_vec - vgs_test))
        if vds_test is None:
            vds_idx = vds_vec.size // 2
        else:
            vds_idx = np.argmin(np.abs(vds_vec - vds_test))
        vbs_idx = np.argmin(np.abs(vbs_vec - vbs_test))

        ss_data = self.get_ss_params(data, self.specs)

        for name in ss_data.signals:
            print(f'{name}: {ss_data[name][eidx, vbs_idx, vds_idx, vgs_idx]:.4g}')


class MOSSPTB(TestbenchManager):
    """Transistor S parameter plotting testbench.  Used mainly for debugging.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'mos_tb_sp')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vbs: float = specs['vbs']
        vgs: float = specs['vgs']
        vds: float = specs['vds']
        is_nmos: bool = specs['is_nmos']
        freq_start: float = specs['freq_start']
        freq_stop: float = specs['freq_stop']
        num_per_dec: int = specs['num_per_dec']
        fix_vbs_sign: bool = specs['fix_vbs_sign']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        freq_num = math.ceil(math.log10(freq_stop) - math.log10(freq_start)) * num_per_dec + 1

        if is_nmos:
            vb_dc = 0.0
        else:
            vds = -vds
            vgs = -vgs
            vb_dc = abs(vgs)

        if fix_vbs_sign:
            if is_nmos:
                vbs = -abs(vbs)
            else:
                vbs = abs(vbs)

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='SP',
                           param='freq',
                           sweep=dict(type='LOG',
                                      start=freq_start,
                                      stop=freq_stop,
                                      num=freq_num,
                                      ),
                           ports=['PORTG', 'PORTD', 'PORTS'],
                           param_type='Y',
                           ),
                      ],
            params=dict(vb_dc=vb_dc, vbs=vbs, vds=vds, vgs=vgs),
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    def print_results(self, data: SimData) -> None:
        data.open_analysis(AnalysisType.SP)
        freq = data['freq']
        y_params = {f'y{a}{b}': data[f'y{a}{b}'][0, :] for a in range(1, 4) for b in range(1, 4)}
        _plot_compare(freq, y_params)


class MOSNoiseTB(TestbenchManager):
    """This class sets up the transistor small-signal noise measurement testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'mos_tb_noise')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vbs_val: Union[float, List[float]] = specs['vbs']
        vgs_min: float = specs['vgs_min']
        vgs_max: float = specs['vgs_max']
        vgs_num: int = specs['vgs_num']
        vds_min: float = specs['vds_min']
        vds_max: float = specs['vds_max']
        vds_num: int = specs['vds_num']
        is_nmos: bool = specs['is_nmos']
        freq_start: float = specs['freq_start']
        freq_stop: float = specs['freq_stop']
        num_per_dec: int = specs['num_per_dec']
        fix_vbs_sign: bool = specs['fix_vbs_sign']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        freq_num = math.ceil(math.log10(freq_stop) - math.log10(freq_start)) * num_per_dec + 1
        vds_start, vds_stop, vb_dc, vbs_list = _fix_vbias_signs(vgs_min, vds_min, vds_max, vbs_val,
                                                                is_nmos, fix_vbs_sign)

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='NOISE',
                           sweep=dict(type='LOG',
                                      start=freq_start,
                                      stop=freq_stop,
                                      num=freq_num,
                                      ),
                           out_probe='VD',
                           ),
                      ],
            params=dict(vb_dc=vb_dc),
            swp_info=[
                ['vbs', dict(type='LIST', values=vbs_list)],
                ['vds', dict(type='LINEAR', start=vds_start, stop=vds_stop, num=vds_num)],
                ['vgs', dict(type='LINEAR', start=vgs_min, stop=vgs_max, num=vgs_num)],
            ],
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def append_integrated_noise(cls, data: SimData, specs: Dict[str, Any], ss_data: SimData,
                                temp: float, fstart: float, fstop: float, scale: float = 1.0
                                ) -> None:
        fg: int = specs['fg']

        data.open_analysis(AnalysisType.NOISE)
        idn = data['out']

        noise_swp_names = data.sweep_params
        cur_points = [data[name] for name in noise_swp_names[1:-1]]
        cur_points.append(np.log(data['freq']))

        # construct new SS parameter result dictionary
        fstart_log = np.log(fstart)
        fstop_log = np.log(fstop)

        # rearrange array axis
        idn = np.log(scale / fg * (idn ** 2))
        delta_list = [1e-6] * (len(noise_swp_names) - 1)
        delta_list[-1] = 1e-3
        integ_noise_list = []
        sim_envs = data.sim_envs
        for idx in range(len(sim_envs)):
            noise_fun = LinearInterpolator(cur_points, idn[idx, ...], delta_list,
                                           extrapolate=True)
            integ_noise_list.append(noise_fun.integrate(fstart_log, fstop_log, axis=-1, logx=True,
                                                        logy=True, raw=True))

        f_delta = fstop - fstart
        gamma = np.array(integ_noise_list) / (4.0 * 1.38e-23 * temp * ss_data['gm'] * f_delta)

        ss_data.insert('gamma', gamma)


class MOSCharSS(MeasurementManager):
    """This class measures small signal parameters of a transistor using Y parameter fitting.

    This measurement is perform as follows:

    1. First, given a user specified current density range, we perform a DC current measurement
       to find the range of vgs needed across corners to cover that range.
    2. Then, we run a S parameter simulation and record Y parameter values at various bias points.
    3. If user specify a noise testbench, a noise simulation will be run at the same bias points
       as S parameter simulation to characterize transistor noise.
    """

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)
        self._ss_data: Optional[SimData] = None
        self._ss_path = self.data_dir / 'ss_params.hdf5'
        meas_specs = self.specs
        self._specs_override = dict(
            fg=meas_specs['fg'],
            is_nmos=meas_specs['is_nmos'],
            vds_min=meas_specs['vds_min'],
            vds_max=meas_specs['vds_max'],
            vds_num=meas_specs['vds_num'],
            fix_vbs_sign=meas_specs.get('fix_vbs_sign', True),
            sim_options=meas_specs.get('sim_options', {}),
        )

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'dc'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:

        # add is_nmos parameter to testbench specification
        tb_name, tb_type, tb_specs, tb_params = super().get_testbench_info(state, prev_output)

        if tb_type == 'dc':
            for key in ['fg', 'is_nmos']:
                tb_specs[key] = self._specs_override[key]
        else:
            tb_specs.update(self._specs_override)

        return tb_name, tb_type, tb_specs, tb_params

    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:

        tb_specs = tb_manager.specs
        if state == 'dc':
            done = False
            next_state = 'sp'
            vgs_vec = MOSIdTB.get_data(data, tb_specs)[0]
            self._specs_override['vgs_min'] = vgs_vec[0]
            self._specs_override['vgs_max'] = vgs_vec[-1]
            self._specs_override['vgs_num'] = vgs_vec.size
            output = dict(vgs=vgs_vec)
        elif state == 'sp':
            testbenches = self.specs['testbenches']
            if 'noise' in testbenches:
                done = False
                next_state = 'noise'
            else:
                done = True
                next_state = ''

            self._ss_data = MOSSSTB.get_ss_params(data, tb_specs)
            if done:
                save_sim_data_hdf5(self._ss_data, self._ss_path)
            output = dict(ss_file=str(self._ss_path))
        elif state == 'noise':
            done = True
            next_state = ''

            temp = self.specs['noise_temp_kelvin']
            fstart = self.specs['noise_integ_fstart']
            fstop = self.specs['noise_integ_fstop']
            scale = self.specs.get('noise_integ_scale', 1.0)

            MOSNoiseTB.append_integrated_noise(self._ss_data, tb_specs, data, temp,
                                               fstart, fstop, scale=scale)
            save_sim_data_hdf5(self._ss_data, self._ss_path)
            output = dict(ss_file=str(self._ss_path))
        else:
            raise ValueError(f'Unknown state: {state}')

        return done, next_state, output


def _fix_vbias_signs(vgs_start: float, vds_min: float, vds_max: float,
                     vbs_val: Union[List[float], float], is_nmos: bool, fix_vbs_sign: bool
                     ) -> Tuple[float, float, float, List[float]]:
    if is_nmos:
        vds_start = vds_min
        vds_stop = vds_max
        vb_dc = 0.0
    else:
        vds_start = -vds_max
        vds_stop = -vds_min
        vb_dc = abs(vgs_start)

    if fix_vbs_sign:
        # handle VBS sign and set parameters.
        if isinstance(vbs_val, list):
            if is_nmos:
                vbs_list = sorted((-abs(v) for v in vbs_val))
            else:
                vbs_list = sorted((abs(v) for v in vbs_val))
        else:
            if is_nmos:
                vbs_list = [-abs(vbs_val)]
            else:
                vbs_list = [abs(vbs_val)]
    else:
        if isinstance(vbs_val, list):
            vbs_list = vbs_val
        else:
            vbs_list = [vbs_val]

    return vds_start, vds_stop, vb_dc, vbs_list


def _plot_compare(fvec: np.ndarray, y_params: Dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    fig, ax_mat = plt.subplots(6, 3, sharex='all', squeeze=False)
    for a in range(3):
        for b in range(3):
            name = f'y{a + 1}{b + 1}'
            ymat = y_params[name]
            ax_real = ax_mat[2 * a, b]
            ax_imag = ax_mat[2 * a + 1, b]
            ax_real.set_ylabel(f'{name} real')
            ax_imag.set_ylabel(f'{name} imag')
            ax_real.semilogx(fvec, ymat.real, 'b+', label='actual')
            ax_imag.semilogx(fvec, ymat.imag, 'b+', label='actual')

            ax_real.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax_imag.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax_real.legend()
            ax_imag.legend()

    plt.show()
