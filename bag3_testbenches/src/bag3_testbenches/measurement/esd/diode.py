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

"""This package contains measurement class for transistors."""

from typing import Tuple, Dict, Any, Sequence, Mapping, Union, List, Optional, Type

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import brentq

from bag.io import read_yaml
from bag.math.dfun import DiffFunction, VectorDiffFunction
from bag.math.interpolate import interpolate_grid
from bag.util.immutable import ImmutableList
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData, AnalysisData, AnalysisType
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5, load_sim_data_hdf5

from ..data.util import brentq_safe


class DiodeDCTB(TestbenchManager):
    """Diode DC characterization testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'diode_tb_sp')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vminus: float = specs['vminus']
        vmin: float = specs['vmin']
        vmax: float = specs['vmax']
        vnum: int = specs['vnum']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='DC',
                           param='vplus',
                           sweep=dict(type='LINEAR',
                                      start=vmin + vminus,
                                      stop=vmax + vminus,
                                      num=vnum,
                                      endpoint=True),
                           ),
                      ],
            params=dict(vminus=vminus),
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def get_data(cls, data: SimData, specs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        vminus: float = specs['vminus']
        vres: float = specs['vres']
        ibias_max: float = specs['ibias_max']

        data.open_analysis(AnalysisType.DC)
        v_dio = data['vp'][0, :] - vminus
        i_dio = -data['VP:p']

        vmax = v_dio[0]
        fun_list = []
        for idx in range(len(data.sim_envs)):
            interp_fun = InterpolatedUnivariateSpline(v_dio, i_dio[idx, :])
            fun_list.append(interp_fun)
            vmax = max(vmax, brentq_safe(interp_fun, v_dio[0], v_dio[-1], ibias_max))

        vmax = math.ceil(vmax / vres) * vres
        num = int(math.ceil((vmax - v_dio[0]) / vres)) + 1
        xvec = np.linspace(v_dio[0], vmax, num, endpoint=True)
        ymat = np.stack([fun(xvec) for fun in fun_list], axis=0)
        return xvec, ymat

    def print_results(self, data: SimData) -> None:
        specs = self.specs
        ibias_max: float = specs['ibias_max']

        xvec, imat = self.get_data(data, specs)

        print(f'for ibias_max={ibias_max:.4g}, vbias_min={xvec[0]:.4g}, vbias_max={xvec[-1]:.4g}')

        import matplotlib.pyplot as plt
        plt.figure()
        for idx, env in enumerate(data.sim_envs):
            plt.plot(xvec, imat[idx, :], label=env)

        plt.legend()
        plt.show()


class DiodeSSTB(TestbenchManager):
    """Diode small-signal parameter extraction testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'diode_tb_sp')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vminus: float = specs['vminus']
        vmin: float = specs['vmin']
        vmax: float = specs['vmax']
        vnum: float = specs['vnum']
        fmin: float = specs['fmin']
        fmax: float = specs['fmax']
        fndec: int = specs['fndec']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        if fndec < 5:
            # we use fndec number of points to determine flatness
            raise ValueError('Insufficient number of points per decade')

        fnum = int(math.ceil(math.log10(fmax) - math.log10(fmin))) * fndec + 1

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='SP',
                           param='freq',
                           sweep=dict(type='LOG',
                                      start=fmin,
                                      stop=fmax,
                                      num=fnum,
                                      endpoint=True),
                           ports=['PORTP'],
                           param_type='Y',
                           ),
                      ],
            params=dict(vminus=vminus),
            swp_info=[('vplus',
                       dict(type='LINEAR', start=vmin, stop=vmax, num=vnum, endpoint=True))],
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def get_ss_params(cls, data: SimData, specs: Dict[str, Any], ibias: Optional[np.ndarray] = None
                      ) -> SimData:
        vminus: float = specs['vminus']
        fndec: int = specs['fndec']
        m_ztol: float = specs.get('m_ztol', 1e-3)
        ctol: float = specs.get('ctol', 1e-19)
        err_rtol: float = specs.get('err_rtol', 0.01)

        data.open_analysis(AnalysisType.SP)
        wvec = data['freq'] * 2 * np.pi
        vbias = data['vplus'] - vminus
        ymat = data['y11']

        if len(ymat.shape) != 3 or not data.is_md:
            raise ValueError('simulation data format incorrect.')

        nenv = len(data.sim_envs)
        nv = vbias.size
        shape = (nenv, nv)
        rs_vec = np.empty(shape)
        rp_vec = np.empty(shape)
        cd_vec = np.empty(shape)
        cp_vec = np.empty(shape)
        err_vec = np.empty(shape)
        viter = range(nv)
        for eidx in range(nenv):
            for vidx in viter:
                rs, rp, cd, cp, rel_err = _get_diode_ss_params(wvec, ymat[eidx, vidx, :],
                                                               fndec, m_ztol, ctol)
                if rel_err > err_rtol:
                    print(f'WARNING: relative fitting error = {rel_err * 100:.2f}% '
                          f'when sim_env={data.sim_envs[eidx]}, vbias={vbias[vidx]:.4g}')
                rs_vec[eidx, vidx] = rs
                rp_vec[eidx, vidx] = rp
                cd_vec[eidx, vidx] = cd
                cp_vec[eidx, vidx] = cp
                err_vec[eidx, vidx] = rel_err

        ss_dict = dict(
            vbias=vbias,
            rs=rs_vec,
            rp=rp_vec,
            cd=cd_vec,
            cp=cp_vec,
            rel_err=err_vec,
        )
        if ibias is not None:
            ss_dict['ibias'] = ibias
        ana_data = AnalysisData(['corners', 'vbias'], ss_dict, True)
        return SimData(data.sim_envs, {'ss': ana_data})

    def print_results(self, data: SimData) -> None:
        specs = self.specs
        vminus: float = specs['vminus']
        vmin: float = specs['vmin']
        vbias_test: float = specs.get('vbias_test', vmin - vminus)
        sim_env_test: str = specs.get('sim_env_test', data.sim_envs[0])

        eidx = data.sim_envs.index(sim_env_test)
        xvec = data['vplus'] - vminus
        vidx = np.argmin(np.abs(xvec - vbias_test))

        ss_data = self.get_ss_params(data, self.specs)

        fvec = data['freq']
        y11 = data['y11'][eidx, vidx, :]

        rs = ss_data['rs'][eidx, vidx]
        rp = ss_data['rp'][eidx, vidx]
        cd = ss_data['cd'][eidx, vidx]
        cp = ss_data['cp'][eidx, vidx]
        _plot_compare(fvec, y11, rs, rp, cd, cp)


class DiodeSPTB(TestbenchManager):
    """Diode S parameter plotting testbench.  Used mainly for debugging.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'diode_tb_sp')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        vminus: float = specs['vminus']
        vplus: float = specs['vplus']
        swp_spec: Mapping[str, Any] = specs['swp_spec']
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='SP',
                           param='freq',
                           sweep=swp_spec,
                           ports=['PORTP'],
                           param_type='Y',
                           ),
                      ],
            params=dict(vminus=vminus, vplus=vplus),
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    def print_results(self, data: SimData) -> None:
        rs: float = self.specs['rs']
        rp: float = self.specs['rp']
        cd: float = self.specs['cd']
        cp: float = self.specs['cp']

        data.open_analysis(AnalysisType.SP)
        freq = data['freq']
        y11 = data['y11'][0, :]

        _plot_compare(freq, y11, rs, rp, cd, cp)


class DiodeCharSS(MeasurementManager):
    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)
        self._vminus: float = 0.0
        self._vplus: Tuple[float, float, int] = (0, 0, 0)
        self._ibias: Optional[np.ndarray] = None

    def get_initial_state(self) -> str:
        """Returns the initial FSM state."""
        return 'dc'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tb_name, tb_type, tb_specs, tb_params = super().get_testbench_info(state, prev_output)
        if state == 'ss':
            tb_specs['vmin'] = self._vplus[0]
            tb_specs['vmax'] = self._vplus[1]
            tb_specs['vnum'] = self._vplus[2]
            tb_specs['vminus'] = self._vminus
        return tb_name, tb_type, tb_specs, tb_params

    def process_output(self, state: str, data: SimData, tb_manager: TestbenchManager
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        tb_specs = tb_manager.specs
        if state == 'dc':
            done = False
            next_state = 'ss'

            xvec, self._ibias = DiodeDCTB.get_data(data, tb_specs)

            self._vminus = tb_specs['vminus']
            self._vplus = xvec[0] + self._vminus, xvec[-1] + self._vminus, xvec.size
            output = dict(vplus=self._vplus)
        elif state == 'ss':
            done = True
            next_state = ''

            ss_data = DiodeSSTB.get_ss_params(data, tb_specs, self._ibias)
            ss_path = self.data_dir / 'ss_params.hdf5'
            save_sim_data_hdf5(ss_data, ss_path)
            output = dict(ss_file=str(ss_path))
        else:
            raise ValueError('Unknown state: %s' % state)

        return done, next_state, output


class DiodeDB:
    def __init__(self, spec_file: Union[str, Path], interp_method: str = 'spline') -> None:
        specs = read_yaml(spec_file)

        root_dir: str = specs['root_dir']

        meas_dir = Path(root_dir)
        ss_data = load_sim_data_hdf5(meas_dir / 'ss_params.hdf5')
        ss_data.open_group('ss')

        self._tot_env_list = ss_data.sim_envs
        self._env_list: Sequence[str] = self._tot_env_list
        self._ss_swp_names = ss_data.sweep_params[1:]
        self._fun_table = self._make_functions(ss_data, interp_method)
        self._ss_outputs = ImmutableList(sorted(self._fun_table.keys()))

    @classmethod
    def _make_functions(cls, ss_data: SimData, interp_method: str) -> Dict[str, List[DiffFunction]]:
        if len(ss_data.sweep_params) != 2 or not ss_data.is_md:
            raise ValueError('simulation data format incorrect.')

        xvec = ss_data[ss_data.sweep_params[1]]
        scale_list = [(xvec[0], xvec[1] - xvec[0])]
        idx_iter = range(len(ss_data.sim_envs))
        return {key: [interpolate_grid(scale_list, ss_data[key][idx, ...], method=interp_method,
                                       extrapolate=True, delta=1e-5) for idx in idx_iter]
                for key in ss_data.signals}

    @property
    def env_list(self) -> Sequence[str]:
        """Sequence[str]: The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list: Sequence[str]) -> None:
        self._env_list = new_env_list

    def get_function_list(self, name: str) -> List[DiffFunction]:
        """Returns a list of functions, one for each simulation environment, for the given output.

        Parameters
        ----------
        name : str
            name of the function.

        Returns
        -------
        output : List[DiffFunction]
            the output vector function.
        """
        master_list = self._fun_table[name]
        return [master_list[self._get_env_index(env)] for env in self._env_list]

    def get_function(self, name: str, env: str = '') -> Union[VectorDiffFunction, DiffFunction]:
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.

        Returns
        -------
        output : Union[VectorDiffFunction, DiffFunction]
            the output vector function.
        """
        if not env and len(self.env_list) == 1:
            env = self.env_list[0]

        if not env:
            return VectorDiffFunction(self.get_function_list(name))
        else:
            return self._fun_table[name][self._get_env_index(env)]

    def query(self, vbias: float, env: str = '') -> Dict[str, np.ndarray]:
        """Query the database for the values associated with the given parameters.

        Either one of vgs and vstar must be specified.  If vds is not specified, we set vds = vgs.
        If vbs is not specified, we set vbs = 0.

        Parameters
        ----------
        vbias : float
            the bias voltage.
        env : str
            If not empty, will return results for this simulation environment only.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        results = {name: self.get_function(name, env=env)(vbias) for name in self._ss_outputs}
        results['vbias'] = vbias
        return results

    def _get_env_index(self, env: str) -> int:
        try:
            return self._env_list.index(env)
        except ValueError:
            raise ValueError(f'environment {env} not found.')


def _get_diode_ss_params(wvec: np.ndarray, yvec: np.ndarray, fndec: int, m_ztol: float, ctol: float,
                         ) -> Tuple[float, float, float, float, float]:
    logw = np.log10(wvec)
    logr = np.log10(yvec.real)

    p0 = np.polyfit(logw[:fndec], logr[:fndec], 1)
    p1 = np.polyfit(logw[-fndec:], logr[-fndec:], 1)

    if p0[0] > m_ztol or p1[0] > m_ztol:
        raise ValueError('real part of Y11 is not flat for a decade at beginning or end '
                         'of frequency range, please expand frequency sweep range.')

    log_g0 = p0[1]
    g0 = 10.0 ** log_g0
    g1 = 10.0 ** p1[1]

    k = 1 - g0 / g1
    kw = k * wvec
    cmax = np.sqrt(g0 * g1 / kw[0] ** 2)
    cmin = g1 / kw[-1]

    avec = kw ** 2 / (g0 * g1)
    bvec = (kw / g1) ** 2

    def deriv_fun(cval):
        tmp1 = 1 + cval ** 2 * avec
        tmp2 = 1 + cval ** 2 * bvec
        lhs = log_g0 - logr + np.log10(tmp1 / tmp2)
        rhs = (avec - bvec) / (tmp1 * tmp2)
        return np.sum(lhs * rhs)

    # noinspection PyTypeChecker
    cd: float = brentq(deriv_fun, cmin, cmax, xtol=ctol)
    rs: float = 1 / g1
    rp: float = 1 / g0 - rs
    yest = 1 / (rs + rp / (1 + 1j * wvec * rp * cd))
    # use least square fitting to find cp
    cp: float = np.linalg.lstsq(wvec.reshape(-1, 1), yvec.imag - yest.imag, rcond=None)[0].item()

    yest_final = yest + 1j * wvec * cp
    rel_err = np.linalg.norm(yvec - yest_final) / (np.linalg.norm(yvec) / yvec.size)
    return rs, rp, cd, cp, rel_err


def _plot_compare(fvec: np.ndarray, y11: np.ndarray, rs: float, rp: float, cd: float, cp: float
                  ) -> None:
    wvec = 2 * np.pi * fvec
    y11_est = 1 / (rs + rp / (1 + 1j * wvec * rp * cd)) + 1j * wvec * cp
    rel_err = np.linalg.norm(y11 - y11_est) / (np.linalg.norm(y11) / y11.size)

    print(f'rs={rs:.4g}, rp={rp:.4g}, cd={cd:.4g}, cp={cp:.4g}, rel_err={rel_err * 100:.2f}%')

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)
    plt.setp(ax0.get_xticklabels(), visible=False)
    ax0.set_ylabel('real')
    ax1.set_ylabel('imag')
    ax0.loglog(fvec, y11.real, 'b+', label='actual')
    ax0.loglog(fvec, y11_est.real, '-r', label='fit')
    ax1.loglog(fvec, y11.imag, 'b+', label='actual')
    ax1.loglog(fvec, y11_est.imag, '-r', label='fit')
    ax0.legend()
    ax1.legend()

    plt.show()
