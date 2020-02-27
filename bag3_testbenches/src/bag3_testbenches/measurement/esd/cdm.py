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

"""This package contains CDM testing related classes."""

from typing import Tuple, Dict, Any, Sequence, Union, Mapping, Type, cast

import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from bag.math import float_to_si_string
from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.simulation.data import (
    SimData, SimNetlistInfo, AnalysisType, netlist_info_from_dict
)
from bag.simulation.base import SimAccess
from bag.simulation.core import TestbenchManager

from ..data.tran import EdgeType, get_first_crossings


def get_rlc_fit(c_slope: float, m: float, i_0: float, req_scale: float, area: float,
                v_cdm: float) -> Dict[str, float]:
    """Compute series RLC parameters in a CDM tester based on fitted parameters.

    Parameters
    ----------
    c_slope : float
        capacitance per square root of area, in F/mm.
    m : float
        resistance fitting slope.
    i_0 : float
        resistance fitting intercept.
    req_scale : float
        resistance scale factor.
    area : float
        die or package area, in mm^2.
    v_cdm : float
        CDM charging voltage.

    Returns
    -------
    params : Dict[str, float]
        dictionary of CDM tester parameters

    Notes
    -----
    inductance is computed assuming a critically damped system
    """
    c_tot = c_slope * math.sqrt(area)
    r_tot = req_scale / (m * math.log(area) + i_0)
    l_tot = 0.25 * r_tot ** 2 * c_tot

    tau_1 = 2 * l_tot / r_tot

    # i_peak for critically damped system
    i_peak = v_cdm / l_tot * tau_1 / math.exp(1)

    return dict(
        c_tot=c_tot,
        r_tot=r_tot,
        l_tot=l_tot,
        t_max=tau_1,
        i_peak=i_peak,
    )


def get_rlc_cal(v_cdm: float, i_peak: float, undershoot: float, tr: float, fwhm: float,
                c_dut: float, c_dg_r: float, rtol: float = 1e-4, max_iter: int = 100
                ) -> Dict[str, float]:
    r"""Compute series RLC parameters in a CDM tester based on calibration data.

    Parameters
    ----------
    v_cdm : float
        CDM charging voltage.
    i_peak : float
        Measured peak CDM current, in Amps.
    undershoot : float
        maximum undershoot current as percentage of peak current.
    tr : float
        10%-90% rise time, in seconds.
    fwhm : float
        full-width-half-maximum of the peak, in seconds.
    c_dut : float
        calibration disk capacitance, in F.
    c_dg_r : float
        free parameter, the ratio c_dg / c_dut
    rtol : float
        root finding relative tolerance.
    max_iter : int
        maximum number of iterations used to find tau_2 upper bound.

    Returns
    -------
    info : Dict[str, float]
        the CDM tester information dictionary.

    Notes
    -----
    Given series RLC, the current waveform is given by:

    .. math::
        :nowrap:

        \begin{align*}
        \tau_1 & = \frac{2L}{R} \\
        \tau_2 & = \left(1/(LC) - \tau_1^{-2}\right)^{-0.5} \\
        I(t) & = \frac{V_0}{L} \cdot e^{-t/\tau_1} \cdot \tau_2 \sin\left(\frac{t}{\tau_2}\right)
        \end{align*}

    Taking the derivative and set equal to 0, we see that the maxima/minima occur at:

    .. math::

        t = \tau_2\left( \arctan\left( \frac{\tau_1}{\tau_2}\right) + n\pi \right)

    Some more math follows and we can solve for RLC values of the CDM tester.
    """
    # compute tau_r = tau_2 / tau_1
    tau_r = -math.log(undershoot) / math.pi

    # solve for tau_2 from fwhm constraint

    # find min/max bound on tau_2
    tau_2_lo = None
    tau_2_min = 4 * tr / 10
    for _ in range(max_iter):
        fwhm_cur = _get_fwhm(tau_2_min, tau_r, rtol)
        if fwhm_cur < fwhm:
            tau_2_lo = tau_2_min
            break
        else:
            tau_2_min /= 2

    if tau_2_lo is None:
        raise RuntimeError(f'max_iter={max_iter} reached and cannot find tau_2 lower bound.')

    tau_2_hi = None
    tau_2_max = 4 * tr * 10
    for _ in range(max_iter):
        fwhm_cur = _get_fwhm(tau_2_max, tau_r, rtol)
        if fwhm_cur > fwhm:
            tau_2_hi = tau_2_max
            break
        else:
            tau_2_max *= 2

    if tau_2_hi is None:
        raise RuntimeError(f'max_iter={max_iter} reached and cannot find tau_2 upper bound.')

    # solve for tau_2 and tau_1
    def fwhm_diff(tau):
        return _get_fwhm(tau, tau_r, rtol) - fwhm

    tau_2 = cast(float, brentq(fwhm_diff, tau_2_lo, tau_2_hi, xtol=fwhm * rtol))
    tau_1 = tau_2 / tau_r
    tr_actual = _get_tr(tau_2, tau_r, rtol)

    # solve for 1/LC
    lc_prod = 1 / (1 / (tau_1 ** 2) + 1 / (tau_2 ** 2))

    # solve for L from Ipeak
    c_dg = c_dut * c_dg_r
    v_init = v_cdm * 1 / (1 + c_dg_r)
    y_max = math.exp(-tau_r * math.atan(1 / tau_r)) / math.sqrt(1 + tau_r ** 2)
    l_tot = v_init / i_peak * tau_2 * y_max

    # solve for R
    r_tot = 2 * l_tot / tau_1

    # solve for Cfg
    c_tot = lc_prod / l_tot
    c_fg = 1 / (1 / (c_tot - c_dg) - 1 / c_dut)

    ans = dict(
        tau_1=tau_1,
        tau_2=tau_2,
        i_peak=i_peak,
        undershoot=undershoot,
        fwhm=_get_fwhm(tau_2, tau_r, rtol),
        v_cdm=v_cdm,
        v_init=v_init,
        f_lc=1 / (lc_prod * 2 * math.pi),
        r_tot=r_tot,
        l_tot=l_tot,
        c_tot=c_tot,
        c_dg=c_dg,
        c_dg_r=c_dg_r,
        c_dut=c_dut,
        c_fg=c_fg,
        tr=tr,
        tr_actual=tr_actual,
        rtol=rtol,
    )

    return ans


def predict_model(params: Dict[str, float]) -> Dict[str, float]:
    c_dut: float = params['c_dut']
    v_cdm: float = params['v_cdm']
    r_tot: float = params['r_tot']
    l_tot: float = params['l_tot']
    c_dg_r: float = params['c_dg_r']
    c_fg: float = params['c_fg']
    rtol: float = params['rtol']

    c_dg = c_dut * c_dg_r

    v_init = v_cdm * 1 / (1 + c_dg_r)
    c_tot = c_dg + (c_dut * c_fg) / (c_dut + c_fg)

    tau_1 = 2 * l_tot / r_tot
    tau_2 = 1 / math.sqrt(1 / (l_tot * c_tot) - 1 / tau_1 ** 2)
    tau_r = tau_2 / tau_1

    theta = math.atan(1 / tau_r)
    y_max = math.exp(-tau_r * theta) / math.sqrt(1 + tau_r ** 2)
    i_peak = v_init / l_tot * tau_2 * y_max
    tr = _get_tr(tau_2, tau_r, rtol)
    fwhm = _get_fwhm(tau_2, tau_r, rtol)
    undershoot = math.exp(-tau_r * np.pi)

    return dict(
        v_init=v_init,
        i_peak=i_peak,
        tr=tr,
        fwhm=fwhm,
        undershoot=undershoot,
    )


class CDMModelTB(TestbenchManager):
    """This class sets up the CDM DUT model testbench.
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'esd_cdm_model')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        params: Dict[str, Union[str, float]] = dict(**specs['sim_params'])
        sim_options: Mapping[str, Any] = specs.get('sim_options', {})

        tsim: float = params['tsim']

        tcharge = _get_tcharge(params)
        exp = int(math.log10(tcharge))
        scale = 10.0 ** exp
        tcharge = math.ceil(tcharge / scale) * scale
        tsim_str = f'tcharge+{float_to_si_string(tsim, self.precision)}'

        params['tcharge'] = tcharge
        params['tsim'] = tsim_str
        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='TRAN',
                           stop='tsim',
                           strobe='tstep',
                           out_start='tcharge',
                           )
                      ],
            params=params,
            swp_info=specs.get('swp_info', []),
            init_voltages=dict(vpin=0.0, vfield=0.0),
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    def print_results(self, data: SimData) -> None:
        results = self.get_current_info(data, self.specs)
        for key, val in results.items():
            val_list = val.flatten().tolist()
            if key == 'tr' or key == 'fwhm':
                val_list = [float_to_si_string(v, self.precision) for v in val_list]
            print(f'{key}: {val_list}')

    @classmethod
    def get_current_info(cls, data: SimData, specs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute output delay from simulation data.

        This method only works for simulation data produced by this TestbenchManager,
        as it assumes a certain clk and input data pattern.

        if the output never resolved correctly, infinity is returned.
        """
        params: Mapping[str, Union[str, float]] = specs['sim_params']

        rmeas: float = params['rmeas']

        data.open_analysis(AnalysisType.TRAN)
        tvec = data['time']
        ivec = data['vmeas'] / rmeas

        ndim = len(ivec.shape)
        ans_shape = ivec.shape[:-1]
        ivec = ivec.reshape((-1, ivec.shape[ndim - 1]))
        ipeak = np.amax(ivec, axis=1)
        itrough = np.abs(np.amin(ivec, axis=1))
        tr = np.zeros(ipeak.shape)
        tw = np.zeros(ipeak.shape)
        for idx in range(ivec.shape[0]):
            ipeak_cur = ipeak[idx]
            yvec = ivec[idx, :]
            t0 = get_first_crossings(tvec, yvec, 0.1 * ipeak_cur, etype=EdgeType.RISE)
            t1 = get_first_crossings(tvec, yvec, 0.9 * ipeak_cur, etype=EdgeType.RISE)
            t2 = get_first_crossings(tvec, yvec, 0.5 * ipeak_cur, etype=EdgeType.RISE)
            t3 = get_first_crossings(tvec, yvec, 0.5 * ipeak_cur, etype=EdgeType.FALL)
            tr[idx] = t1 - t0
            tw[idx] = t3 - t2

        return dict(
            ipeak=ipeak.reshape(ans_shape),
            undershoot=(itrough / ipeak * 100).reshape(ans_shape),
            tr=tr.reshape(ans_shape),
            fwhm=tw.reshape(ans_shape),
        )


class CDMRLCTB(TestbenchManager):
    """A CDM verification testbench using lumped/measured RLC parameters
    """

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        # noinspection PyTypeChecker
        return ModuleDB.get_schematic_class('bag3_testbenches', 'esd_cdm_rlc_tb_tran')

    def get_netlist_info(self) -> SimNetlistInfo:
        specs = self.specs
        rtot: float = specs['rtot']
        ltot: float = specs['ltot']
        ctot: float = specs['ctot']
        tsim: float = specs['tsim']
        td: float = specs['td']
        tr: float = specs['tr']
        vcdm: float = specs['vcdm']

        sim_options: Mapping[str, Any] = specs.get('sim_options', {})
        tran_options: Mapping[str, Any] = specs.get('tran_options', {})

        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='TRAN',
                           stop='tsim',
                           strobe='tstep',
                           out_start='tcharge',
                           options=tran_options,
                           )
                      ],
            params=dict(
                rtot=rtot,
                ltot=ltot,
                ctot=ctot,
                tsim=tsim,
                td=td,
                tr=tr,
                vcdm=vcdm,
            ),
            # disable sweeping
            swp_info=[],
            options=sim_options,
        )

        return netlist_info_from_dict(sim_setup)

    def print_results(self, data: SimData) -> None:
        time = data['time']
        i_cdm = data['IM:in']
        v_pin = data['vpin']
        v_sup = data['VDD']

        _print_max(time, i_cdm, 'i_max', 'A')
        _print_max(time, v_pin, 'v_pin_max', 'V')
        _print_max(time, v_sup, 'v_sup_max', 'V')


def _print_max(xvec: np.ndarray, yvec: np.ndarray, name: str, unit: str) -> None:
    max_idx = np.argmax(np.abs(yvec))
    t_max = xvec[max_idx]
    y_max = yvec[max_idx]
    print(f'{name} = {float_to_si_string(y_max)} {unit} at t = {float_to_si_string(t_max)} s')


def _get_fwhm(tau_2: float, tau_r: float, rtol: float) -> float:
    theta = math.atan(1 / tau_r)
    y_max = math.exp(-tau_r * theta) / math.sqrt(1 + tau_r ** 2)
    t_max = tau_2 * theta
    t_zero = tau_2 * math.pi
    tau_1 = tau_2 / tau_r

    args = (tau_1, tau_2, 0.5 * y_max)
    xtol = t_max * rtol
    t0 = cast(float, brentq(_exp_sin, 0, t_max, args=args, xtol=xtol))
    t1 = cast(float, brentq(_exp_sin, t_max, t_zero, args=args, xtol=xtol))

    return t1 - t0


def _get_tr(tau_2: float, tau_r: float, rtol: float) -> float:
    theta = math.atan(1 / tau_r)
    y_max = math.exp(-tau_r * theta) / math.sqrt(1 + tau_r ** 2)
    t_max = tau_2 * theta
    tau_1 = tau_2 / tau_r

    xtol = t_max * rtol
    t0 = cast(float, brentq(_exp_sin, 0, t_max, args=(tau_1, tau_2, 0.1 * y_max), xtol=xtol))
    t1 = cast(float, brentq(_exp_sin, 0, t_max, args=(tau_1, tau_2, 0.9 * y_max), xtol=xtol))

    return t1 - t0


def _exp_sin(t: float, tau_1: float, tau_2: float, offset: float) -> float:
    return math.exp(-t / tau_1) * math.sin(t / tau_2) - offset


def _get_tcharge(params: Mapping[str, Union[str, float]]) -> float:
    num_tau = 5

    rcharge: float = params['rcharge']
    cdut: float = params['cdut']
    cdg: float = params['cdg']
    cfg: float = params['cfg']

    ccharge = cfg + (cdut * cdg) / (cdut + cdg)

    tcharge = rcharge * ccharge * num_tau
    exp = int(math.log10(tcharge))
    scale = 10.0 ** exp
    tcharge = math.ceil(tcharge / scale) * scale

    return tcharge
