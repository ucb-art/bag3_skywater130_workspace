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

"""This package contains transient testbenches for comparators."""

from typing import Optional, Tuple, Dict, Any, Union, cast, Mapping

from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.optimize import brentq, curve_fit

from bag.simulation.data import SimData
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.math.interpolate import LinearInterpolator
from bag.math import float_to_si_string
from bag.util.search import FloatBinaryIterator
from bag.util.immutable import Param
from bag.io.file import open_file
from bag.concurrent.util import GatherHelper

from ..tran.digital import DigitalTranTB


@dataclass
class StepWave:
    """A data class to generate ramp pwl waveform in discrete steps"""
    cmode: float
    ampl: float
    res: float
    nper: int
    tper: float
    trf: float
    t_rst: float
    tsim: float = None
    nlevels: int = None
    fname: Path = ''

    def __post_init__(self):
        self.nlevels = int(self.ampl / self.res) * 2 + 1
        self.tsim = (2 * self.nlevels - 1) * (self.nper * self.tper)

    def create_file(self, pwl_fname: Path) -> None:
        v_list = np.linspace(self.cmode - self.ampl, self.cmode + self.ampl, self.nlevels,
                             endpoint=True)
        time = self.t_rst
        per = self.nper * self.tper
        with open_file(pwl_fname, 'w') as f:
            f.writelines(f'{time} {v_list[0]}\n')
            for v in v_list:
                f.writelines(f'{time + self.trf + self.tper / 4} {v}\n')
                time += per
                f.writelines(f'{time} {v}\n')
            for v in reversed(v_list[:-1]):
                f.writelines(f'{time + self.trf + self.tper / 4} {v}\n')
                time += per
                f.writelines(f'{time} {v}\n')
        self.fname = pwl_fname


@dataclass
class ImpulseWave:
    """A data class to generate pwl waveform for impulse response test"""
    cmode: float
    t_width: float
    v_pul: float
    tsim: float
    impulse_delay: float
    tper: float
    tr: float
    t_pul: float = None
    fname: Path = ''

    def __post_init__(self):
        self.t_pul = self.tper / 4 - self.tr / 2 + self.impulse_delay

    def create_file(self, pwl_fname: Path) -> None:
        ini_ampl = self.cmode
        pul_ampl = ini_ampl + self.v_pul
        with open_file(pwl_fname, 'w') as f:
            f.writelines(f'{0.0} {ini_ampl}\n')
            time = self.t_pul
            f.writelines(f'{time} {ini_ampl}\n')
            f.writelines(f'{time + self.t_width / 50} {pul_ampl}\n')
            time += self.t_width
            f.writelines(f'{time} {pul_ampl}\n')
            f.writelines(f'{time + self.t_width / 50} {ini_ampl}\n')
        self.fname = pwl_fname


class CompTranTB(DigitalTranTB):
    """This class sets up the comparator transient measurement testbench.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tsim = None
        super().__init__(*args, **kwargs)

    @classmethod
    def plot_waveforms(cls, sim_data: SimData) -> None:
        time = sim_data['time']
        Vin = sim_data['v_SIG'][0]
        Vref = sim_data['v_REF'][0]
        int_ref = sim_data['int_ref'][0]
        rstb = sim_data['RSTB'][0]
        rstd = sim_data['RSTD'][0]
        rst = sim_data['RST'][0]
        VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]

        # i_Vref = sim_data['VCM:p'][0]

        VsampP = sim_data['XDUT_wrap.v_SAMP'][0]
        # VsampM = sim_data['XDUT_wrap.v_SAMPM'][0]
        VacP = sim_data['v_ACP'][0]
        VacM = sim_data['v_ACM'][0]
        VintP, VintM = sim_data['v_SWP'][0], sim_data['v_SWM'][0]
        VcompP, VcompM = sim_data['v_COMPP'][0], sim_data['v_COMPM'][0]

        mode = 0

        if mode:
            plt.subplot(5, 1, 1)
            plt.plot(time, rstb)
            # plt.plot(time, clkd)
            plt.ylabel('Clock')

            plt.subplot(5, 1, 2)
            plt.plot(time, Vin)
            plt.plot(time, Vref)
            plt.ylabel('Input')

            plt.subplot(5, 1, 3)
            plt.plot(time, VsampP)
            # plt.plot(time, VacP)
            plt.ylabel('Sampled, AC Cap')

            plt.subplot(5, 1, 4)
            plt.plot(time, VcompP)
            plt.plot(time, VcompM)
            plt.ylabel('Comp Output')

            plt.subplot(5, 1, 5)
            plt.plot(time, VoP)
            plt.plot(time, VoM)
            plt.ylabel('Final Output')

        else:
            # plt.subplot(211)
            plt.plot(time, rstb, label='RSTB')
            plt.plot(time, rst, label='RST')
            plt.plot(time, rstd, label='RSTD')
            plt.plot(time, Vin, label='inp')
            plt.plot(time, Vref, label='Vref')
            plt.plot(time, int_ref, label='int_ref')
            plt.plot(time, VsampP, label='sampp')
            # plt.plot(time, VsampM, label='sampm')
            plt.plot(time, VacP, label='acp')
            plt.plot(time, VacM, label='acm')
            plt.plot(time, VcompP, label='compp')
            plt.plot(time, VcompM, label='compm')
            # plt.plot(time, VintP, label='intp')
            # plt.plot(time, VintM, label='intm')
            plt.plot(time, VoP, label='outp')
            plt.plot(time, VoM, label='outm')
            plt.legend()
            # plt.subplot(212)
            # plt.plot(time, i_Vref, label='i_Vref')
            # plt.legend()

        plt.show()

    def get_switching_thres(self, sim_data: SimData, vdd: float) -> Dict[str, Any]:
        """ Post processing function for getting switching threshold of comparator from transient
        simulation using ramp input waveform

        Parameters
        ----------
        sim_data : SimData
            the simulation data
        vdd : float
            supply voltage

        Returns
        -------
        Dict of switching threshold, hysteresis and error bound
        """

        sim_params: Param = self.specs['sim_params']
        tper = sim_params['tper']
        t_rst = sim_params['t_rst']

        Vin = sim_data['v_SIG'][0]
        VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]
        VcompP, VcompM = sim_data['v_COMPP'][0], \
                         sim_data['v_COMPM'][0]
        VacP = sim_data['v_ACP'][0]
        time = sim_data['time']
        samp_time = np.arange(start=t_rst + tper / 4 + tper, stop=time[-1], step=tper)
        delta_list = [1e-12]

        fp = LinearInterpolator([time], VoP, delta_list)
        fm = LinearInterpolator([time], VoM, delta_list)
        inp = LinearInterpolator([time], Vin, delta_list)

        fcompP = LinearInterpolator([time], VcompP, delta_list)
        fcompM = LinearInterpolator([time], VcompM, delta_list)
        facP = LinearInterpolator([time], VacP, delta_list)

        samp_VoP, samp_VoM = fp(samp_time), fm(samp_time)
        dig_VoP, dig_VoM = (samp_VoP > vdd / 2).astype(int), (samp_VoM > vdd / 2).astype(int)
        assert np.all(dig_VoM == 1 - dig_VoP), "Check waveforms as output may be metastable"

        # find times when output flipped
        flip_detect = np.logical_xor(dig_VoP[:-1], dig_VoP[1:])
        flip_idx_list = np.where(flip_detect == 1)[0]
        flip_time = samp_time[flip_idx_list]
        sw_thres = []
        for t in flip_time:
            sw_thres.append(inp([t - 3 * tper / 4, t + tper / 4]))

        if len(sw_thres) == 2:
            # calculate offset, hysteresis, error bounds
            # low to high
            thres01 = (sw_thres[0][0] + sw_thres[0][1]) / 2
            err01 = abs(sw_thres[0][1] - thres01)
            # high to low
            thres10 = (sw_thres[1][0] + sw_thres[1][1]) / 2
            err10 = abs(sw_thres[1][1] - thres10)

            mean_thres = (thres01 + thres10) / 2

            output = dict(
                mean_thres=float(mean_thres),
                hysteresis=float(thres01-mean_thres),
                err_bound=float(err01+err10),
            )
        else:
            output = dict(
                sw_points=sw_thres,
            )
        return output

    def check_overdrive_output(self, sim_data: SimData, sign: float, num_rst: int, num_eval: int
                               ) -> bool:
        """ This helper function checks if the differential comparator output (after output stage)
        is true differential digital, or if there are metastability issues

        Parameters
        ----------
        sim_data : SimData
            the simulation data
        sign : float
            1.0 if differential input transitions from large negative to small positive
            -1.0 if differential input transitions from large positive to small negative
        num_rst : int
            number of periods where input is large negative / positive for resetting
        num_eval : int
            number of periods where input is small positive / negative for evaluation

        Returns
        -------
        True if output is true differential
        """

        sim_params: Param = self.specs['sim_params']
        tper = sim_params['tper']
        t_rst = sim_params['t_rst']
        vdd = sim_params['vdd']

        VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]
        time = sim_data['time']
        tstart = t_rst + tper / 4 + tper
        tstop = t_rst + tper / 4 + (num_rst + num_eval) * tper
        assert tstop < time[-1], 'Check simulation time'
        samp_time = np.linspace(tstart, tstop, num_rst + num_eval)
        delta_list = [1e-12]

        fp = LinearInterpolator([time], VoP, delta_list)
        fm = LinearInterpolator([time], VoM, delta_list)

        samp_VoP, samp_VoM = fp(samp_time), fm(samp_time)
        dig_VoP, dig_VoM = (samp_VoP > vdd / 2).astype(int), (samp_VoM > vdd / 2).astype(int)
        assert np.all(dig_VoM == 1 - dig_VoP), "Check waveforms as output may be metastable"
        if sign > 0:
            expected = np.concatenate((np.zeros(num_rst), np.ones(num_eval))).astype(int)
        else:
            expected = np.concatenate((np.ones(num_rst), np.zeros(num_eval))).astype(int)

        return np.all(dig_VoP == expected)

    def calc_ones(self, sim_data: SimData, num_rst: int, num_points: int) -> float:
        """ This helper function is used in overdrive noise measurement manager to estimate standard
        deviation of input referred noise by calculating number of 'ones' at comparator output for
        'num_points' number of evaluation points
        Parameters
        ----------
        sim_data : SimData
            the simulation data
        num_rst : int
            number of periods where input is large negative / positive for resetting
        num_points : int
            number of evaluation points

        Returns
        -------
        percentage of 'ones' at comparator output
        """
        sim_params: Param = self.specs['sim_params']
        tper = sim_params['tper']
        vdd = sim_params['vdd']
        t_rst = sim_params['t_rst']

        VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]
        time = sim_data['time']
        tstart = t_rst + tper / 4 + (num_rst + 1) * tper
        tstop = t_rst + tper / 4 + (num_rst + 1) * num_points * tper
        assert tstop < time[-1], 'Check simulation time'
        samp_time = np.linspace(tstart, tstop, num_points)
        delta_list = [1e-12]

        fp = LinearInterpolator([time], VoP, delta_list)
        fm = LinearInterpolator([time], VoM, delta_list)

        samp_VoP, samp_VoM = fp(samp_time), fm(samp_time)
        dig_VoP, dig_VoM = (samp_VoP > vdd / 2).astype(int), (samp_VoM > vdd / 2).astype(int)
        assert np.all(dig_VoM == 1 - dig_VoP), "Check waveforms as output may be metastable"

        return sum(dig_VoP) / num_points

    def get_taus(self, sim_data0: SimData, sim_data1: SimData, num_rst: int, sign: float,
                 plot_flag: bool = False) -> Tuple[float, float]:
        """This helper function is used to get the regeneration and resettling time constants

        Parameters
        ----------
        sim_data0 : SimData
            the first simulation data (for evaluation amplitude ampl0)
        sim_data1 : SimData
            the first simulation data (for evaluation amplitude ampl1)
        num_rst : int
            number of periods where input is large negative / positive for resetting
        sign : float
            1.0 if differential input transitions from large negative to small positive
            -1.0 if differential input transitions from large positive to small negative
        plot_flag : bool
            True to plot data for debugging

        Returns
        -------
        regeneration time constant, resettling time constant
        """
        sim_params = self.specs['sim_params']
        vdd = sim_params['vdd']
        tper = sim_params['tper']
        td = sim_params['td']
        trf = sim_params['trf']
        t_rst = sim_params['t_rst']

        time0 = sim_data0['time']
        compM0 = sim_data0['v_COMPM'][0]
        compP0 = sim_data0['v_COMPP'][0]
        rstd = sim_data0['RSTD'][0]
        time1 = sim_data1['time']
        compM1 = sim_data1['v_COMPM'][0]
        compP1 = sim_data1['v_COMPP'][0]

        fM0 = LinearInterpolator([time0], compM0, [1e-12])
        fP0 = LinearInterpolator([time0], compP0, [1e-12])

        fM1 = LinearInterpolator([time1], compM1, [1e-12])
        fP1 = LinearInterpolator([time1], compP1, [1e-12])

        # resample to have uniform time axis
        time = np.linspace(min(time0[0], time1[0]), max(time0[-1], time1[-1]),
                           num=max(len(time0), len(time1)))

        # --- Regeneration --- #
        self.log('Curve fitting for regeneration')
        # find correct time window
        idx0 = np.where(time - (t_rst + (num_rst + 0.25) * tper + td + trf) < 0)[0][-1]
        idx1 = np.where(time - (t_rst + (num_rst + 0.75) * tper + td + trf) < 0)[0][-1]
        time_win = time[idx0:idx1]
        compM0 = fM0(time_win)
        compP0 = fP0(time_win)
        rstd = LinearInterpolator([time0], rstd, [1e-12])(time_win)

        compM1 = fM1(time_win)
        compP1 = fP1(time_win)

        # reduce time window further for fitting
        # 1. Start when reset_delay goes down to vdd / 2
        idx_ini = np.where(rstd > vdd / 2)[0][-1]

        # 2. Final index for compP
        compP_diff = compP1 - compP0
        if sign > 0:
            val = np.max(compP_diff)
            idx_finP = np.where(compP_diff > 0.75 * val)[0][0]
        else:
            val = np.min(compP_diff)
            idx_finP = np.where(compP_diff < 0.75 * val)[0][0]

        # 3. Final index for compM
        compM_diff = compM1 - compM0
        if sign > 0:
            val = np.min(compM_diff)
            idx_finM = np.where(compM_diff < 0.75 * val)[0][0]
        else:
            val = np.max(compM_diff)
            idx_finM = np.where(compM_diff > 0.75 * val)[0][0]

        idx_fin = min(idx_finP, idx_finM)
        idx_dict = dict(
            idx_inip=idx_ini,
            idx_inim=idx_ini,
            idx1=idx_finP,
            idx2=idx_fin,
            trP=time_win[idx_fin] - time_win[idx_ini],
            trM=time_win[idx_fin] - time_win[idx_ini],
        )

        tau_rgn = self.compute_tau(time_win, compP_diff, compM_diff, 'rgn', idx_dict,
                                   plot_flag=plot_flag)

        # --- Reset --- #
        self.log('Curve fitting for resettling')
        # find correct time window
        idx2 = np.where(time - (t_rst + (num_rst + 1.25) * tper + td + trf) < 0)[0][-1]
        time_win2 = time[idx1:idx2]
        compM2 = fM0(time_win2)
        compP2 = fP0(time_win2)

        tau_rst = self.compute_tau(time_win2, compP2, compM2, 'rst', {}, plot_flag=plot_flag)

        if plot_flag:
            plt.subplot(411)
            plt.plot(time_win, compP0, label='eval_ampl0')
            plt.plot(time_win, compP1, label='eval_ampl1')
            plt.plot(time_win, rstd, label='rstd')
            plt.axvline(x=time_win[idx_ini])
            plt.axvline(x=time_win[idx_finP])
            plt.legend()
            plt.title('v_COMPP')
            plt.subplot(412)
            plt.plot(time_win, compP1 - compP0, label='diff')
            plt.axvline(x=time_win[idx_ini])
            plt.axvline(x=time_win[idx_finP])
            plt.legend()
            plt.title('v_COMPP')
            plt.subplot(413)
            plt.plot(time_win, compM0, label='eval_ampl0')
            plt.plot(time_win, compM1, label='eval_ampl1')
            plt.plot(time_win, rstd, label='rstd')
            plt.axvline(x=time_win[idx_ini])
            plt.axvline(x=time_win[idx_finM])
            plt.legend()
            plt.title('v_COMPM')
            plt.subplot(414)
            plt.plot(time_win, compM1 - compM0, label='diff')
            plt.axvline(x=time_win[idx_ini])
            plt.axvline(x=time_win[idx_finM])
            plt.legend()
            plt.title('v_COMPM')

            plt.tight_layout()
            plt.show()
        return tau_rgn, tau_rst

    def get_times(self, sim_data: SimData, num_rst: int) -> Tuple[float, float]:
        """This helper function gets the 10-90% time for a regeneration / resettling transition

        Parameters
        ----------
        sim_data : SimData
            the simulation data
        num_rst : int
            number of periods where input is large negative / positive for resetting

        Returns
        -------
        10 - 90% regeneration time, 10 - 90% resettling time
        """
        sim_params = self.specs['sim_params']
        tper = sim_params['tper']
        trf = sim_params['trf']
        td = sim_params['td']
        t_rst = sim_params['t_rst']

        time = sim_data['time']
        compM = sim_data['v_COMPM'][0]
        compP = sim_data['v_COMPP'][0]

        # compress time and sig arrays to one rise/fall edge, then call compute_tset
        # regeneration
        idx0 = np.where(time - (t_rst + (num_rst + 0.25) * tper + td + trf) < 0)[0][-1]
        idx1 = np.where(time - (t_rst + (num_rst + 0.75) * tper + td + trf) < 0)[0][-1]
        _, _, trP = self.get_set_idx(time[idx0:idx1], compP[idx0:idx1])
        _, _, trM = self.get_set_idx(time[idx0:idx1], compM[idx0:idx1])
        t_rgn = max(trP, trM)

        # reset
        idx2 = np.where(time - (t_rst + (num_rst + 1.25) * tper + td + trf) < 0)[0][-1]
        _, _, trP = self.get_set_idx(time[idx1:idx2], compP[idx1:idx2])
        _, _, trM = self.get_set_idx(time[idx1:idx2], compM[idx1:idx2])
        t_rst = max(trP, trM)

        return t_rgn, t_rst

    def compute_tau(self, time: np.ndarray, sigP: np.ndarray, sigM: np.ndarray, mode: str,
                    idx_dict: Optional[Dict[str, Union[int, float]]] = None,
                    plot_flag: bool = False) -> float:
        """This helper function computes the worst case time constant for regeneration /
        resettling among the differential comparator outputs (before output stage)

        Parameters
        ----------
        time : np.ndarray
            the time axis
        sigP : np.ndarray
            signal at plus terminal
        sigM : np.ndarray
            signal at minus terminal
        mode : str
            'rgn' for curve fitting to regeneration equation
            'rst' for curve fitting to resettling equation
        idx_dict : Optional[Dict[str, Union[int, float]]]
            Optional, pre-computed time windows with single transition
        plot_flag : bool
            True to plot waveforms for debugging

        Returns
        -------
        time constant
        """
        if idx_dict:
            # get time markers from sigP
            idx_inip, idx1, _ = idx_dict['idx_inip'], idx_dict['idx1'], idx_dict['trP']

            # get end time marker from sigM
            idx_inim, idx2, _ = idx_dict['idx_inim'], idx_dict['idx2'], idx_dict['trM']
        else:
            # get time markers from sigP
            idx_inip, idx1, _ = self.get_set_idx(time, sigP)

            # get end time marker from sigM
            idx_inim, idx2, _ = self.get_set_idx(time, sigM)

        idx0 = max(idx_inip, idx_inim)

        # fit exponential to sigP and get tauP
        def funP(t, t0, tau):
            if mode == 'rgn':
                return sigP[idx0] * np.exp((t - t0) / tau)
            if mode == 'rst':
                return sigP[idx0] + (sigP[idx1] - sigP[idx0]) * (1 - np.exp((t0 - t) / tau))
            raise ValueError(f'Unknown mode = {mode}')

        varP, covP = curve_fit(funP, time[idx0:idx1], sigP[idx0:idx1],
                               p0=(time[idx0], (time[idx1] - time[idx0]) / np.log(9)))

        # fit exponential to sigM and get tauM
        def funM(t, t0, tau):
            if mode == 'rgn':
                return sigM[idx0] * np.exp((t - t0) / tau)
            if mode == 'rst':
                return sigM[idx0] + (sigM[idx1] - sigM[idx0]) * (1 - np.exp((t0 - t) / tau))
            raise ValueError(f'Unknown mode = {mode}')

        varM, covM = curve_fit(funM, time[idx0:idx2], sigM[idx0:idx2],
                               p0=(time[idx0], (time[idx2] - time[idx0]) / np.log(9)))

        if plot_flag:
            plt.subplot(211)
            val = 0
            plt.plot(time[idx0 - val:idx2 + val], sigP[idx0 - val:idx2 + val], label='meas')
            plt.plot(time[idx0 - val:idx2 + val], funP(time[idx0 - val:idx2 + val], *varP),
                     label='fit')
            plt.legend()
            plt.title('v_COMPP')
            plt.subplot(212)
            plt.plot(time[idx0 - val:idx2 + val], sigM[idx0 - val:idx2 + val], label='meas')
            plt.plot(time[idx0 - val:idx2 + val], funM(time[idx0 - val:idx2 + val], *varM),
                     label='fit')
            plt.legend()
            plt.title('v_COMPM')

            plt.tight_layout()
            plt.show()
        return max(varP[1], varM[1])

    @classmethod
    def get_set_idx(cls, time: np.ndarray, sig: np.ndarray, plot_flag: bool = False
                    ) -> Tuple[float, float, float]:
        """
        This function computes time taken to settle from initial value of sig to final value of sig

        Parameters
        ----------
        time : np.ndarray
            time axis
        sig : np.ndarray
            signal values corresponding to time axis
        plot_flag : bool
            True to plot sig vs time for debugging

        Returns
        -------
        Time for 10 % - 90 % settling
        """
        if sig[0] > sig[-1]:
            sig = - sig
        err = np.abs((sig[-1] - sig[0]) * 0.1)  # 10% of difference between initial and final value

        sig_diff = np.diff(sig)
        if np.abs(sig_diff[-1]) > 0.1:
            raise ValueError('Check waveform; waveform has not settled in given time window')

        # find last index where sig is below final val - error
        idx = np.where(sig - (sig[-1] - err) < 0)[0][-1]
        fun = interp.InterpolatedUnivariateSpline(time, sig - (sig[-1] - err))
        tfinal = cast(float, brentq(fun, time[idx], time[idx + 1]))

        # check for overshoot
        idx2_list = np.where(sig - (sig[-1] + err) > 0)[0]
        if idx2_list.size != 0:
            idx2 = idx2_list[-1]
            fun2 = interp.InterpolatedUnivariateSpline(time, sig - (sig[-1] + err))
            tfinal2 = cast(float, brentq(fun2, time[idx2], time[idx2 + 1]))
            tfinal = max(tfinal, tfinal2)
            idx = max(idx, idx2)

        # find last index where sig is below initial val + error
        idx3 = np.where(sig - (sig[0] + err) < 0)[0][-1]
        fun3 = interp.InterpolatedUnivariateSpline(time, sig - (sig[0] + err))
        tini = cast(float, brentq(fun3, time[idx3], time[idx3 + 1]))

        if plot_flag:
            plt.figure()
            plt.plot(time, sig)
            plt.axvline(x=tini)
            plt.axvline(x=tfinal)
            plt.show()

        return idx3, idx, tfinal - tini


class RampMM(MeasurementManager):
    """This class finds comparator switching threshold using ramp input waveform.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """Setup parallel processing"""
        helper = GatherHelper()

        # Setup parallel simulations and measurements
        corners = self.specs['corners']
        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            for vdd in vdd_list:
                helper.append(self.meas_cor_vdd(name, sim_dir, sim_db, dut, env, vdd))

        coro_results = await helper.gather_err()
        results = {}

        # Construct final results dictionary
        idx = 0
        keys = ['mean_thres', 'hysteresis', 'err_bound']

        fig = plt.figure()
        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            results[env] = dict(vdd=vdd_list)
            results[env].update({key: [] for key in keys})
            for vdd in vdd_list:
                for key in keys:
                    results[env][key].append(coro_results[idx][key])
                idx += 1

        return results

    async def meas_cor_vdd(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                           dut: Optional[DesignInstance], env: str, vdd: float) -> Dict[str, Any]:
        """Run measurement per corner and supply voltage"""
        # --- Ramp test --- #
        sim_id = 'ramp'
        sim_dir = sim_dir / env / float_to_si_string(vdd)
        tbm_specs = self._get_tbm_specs(sim_dir / sim_id, env, vdd)

        tbm = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm_specs))
        sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id, dut, tbm, {})
        results = tbm.get_switching_thres(sim_results.data, vdd)

        return results

    def _get_tbm_specs(self, sim_dir: Path, env: str, vdd: float) -> Dict[str, Any]:
        specs = self.specs
        const_params = specs['const_params']

        # create PWL file
        r_wave = specs['r_wave_params']
        step_wave = StepWave(**r_wave)
        pwl_fname = 'pwl_ramp.txt'
        sim_dir.mkdir(parents=True, exist_ok=True)
        step_wave.create_file(sim_dir / pwl_fname)

        # create RST, RSTD, RSTB, v_REF, v_SIG signals
        pulse_list = [dict(pin='RST', tper='tper', tpw='tper/2', trf='trf'),
                      dict(pin='RSTD', tper='tper', tpw='tper/2', trf='trf', td='td'),
                      dict(pin='RSTB', tper='tper', tpw='tper/2', trf='trf', pos=False)]
        pwr_domain = dict(RST=('VSS', 'VDD'),
                          RSTD=('VSS', 'VDD'),
                          RSTB=('VSS', 'VDD'),
                          int_in=('VSS', 'VDD'),
                          v_SIG=('VSS', 'VDD'),
                          v_REF=('VSS', 'VDD'),
                          v_BIAS=('VSS', 'VDD'),
                          v_OUTP=('VSS', 'VDD'),
                          v_OUTM=('VSS', 'VDD'),
                          )

        # add resistor and capacitors and vpwlf in TB
        load_list = [dict(pin='int_in', nin='v_SIG', type='res', value='rin'),
                     dict(pin='int_in', type='vpwlf', value=pwl_fname),
                     dict(pin='int_ref', nin='v_REF', type='res', value='rref'),
                     dict(pin='int_bias', nin='v_BIAS', type='res', value='rbias'),
                     dict(pin='v_SIG', type='cap', value='cin'),
                     dict(pin='v_REF', type='cap', value='cref'),
                     dict(pin='v_BIAS', type='cap', value='cbias'),
                     dict(pin='v_OUTP', type='cap', value='cload'),
                     dict(pin='v_OUTM', type='cap', value='cload'),
                     dict(pin='v_SAMP', nin='v_ACP', type='cap', value='cpre'),
                     dict(pin='v_COMPM', nin='v_ACM', type='cap', value='cpre'),
                     dict(pin='v_SWP', nin='v_COMPP', type='cap', value='cc'),
                     dict(pin='v_SWM', nin='v_COMPM', type='cap', value='cc')]

        tbm_specs = dict(
            dut_pins=['VDD', 'VSS', 'RST', 'RSTD', 'RSTB', 'RX_EN', 'v_BIAS', 'v_SIG',
                      'v_REF', 'v_COMPM', 'v_COMPP', 'v_OUTM', 'v_OUTP', 'v_SWM', 'v_SWP',
                      'v_ACM', 'v_ACP', 'v_SAMP'],
            pulse_list=pulse_list,
            pwr_domain=pwr_domain,
            load_list=load_list,
            sup_values=dict(VSS=0,
                            VDD=vdd,
                            RX_EN=vdd,
                            int_ref=step_wave.cmode,
                            int_bias=const_params['vbias'],
                            ),
            sim_envs=[env],
            sim_params=dict(
                t_sim=step_wave.tsim,
                **specs['sim_params']
            ),
            sim_options=specs['sim_options'],
            tran_options=specs['tran_options'],
        )
        return tbm_specs


class OverdriveMM(MeasurementManager):
    """This class performs measurements based on the overdrive recovery test"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        raise RuntimeError('Unused')

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:
        raise RuntimeError('Unused')

    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance]) -> Dict[str, Any]:
        """Setup parallel processing"""
        helper = GatherHelper()

        # Setup parallel simulations and measurements
        corners = self.specs['corners']
        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            for vdd in vdd_list:
                helper.append(self.meas_cor_vdd(name, sim_dir, sim_db, dut, env, vdd))

        coro_results = await helper.gather_err()
        results = {}

        # Construct final results dictionary
        estimate_tau = self.specs['estimate_tau']

        idx = 0
        keys = ['mean_thres', 'hysteresis', 't_rgn', 't_rst']
        if estimate_tau:
            keys += ['tau_rgn', 'tau_rst']

        fig = plt.figure()
        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            results[env] = dict(vdd=vdd_list)
            results[env].update({key: [] for key in keys})
            for vdd in vdd_list:
                for key in keys:
                    results[env][key].append(coro_results[idx][key])
                idx += 1
            if len(vdd_list) > 1:
                plt.plot(results[env]['vdd'], results[env]['mean_thres'], label=env, marker='o')
        if fig.get_axes():
            plt.xlabel('VDD (in V)')
            plt.ylabel('Switching threshold (in V)')
            plt.legend()
            plt.show()

        return results

    async def meas_cor_vdd(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                           dut: Optional[DesignInstance], env: str, vdd: float) -> Dict[str, Any]:
        """Run measurement per corner and supply voltage"""
        tbm_specs = self._get_tbm_specs(env, vdd)
        sim_dir = sim_dir / env / float_to_si_string(vdd)

        # --- Overdrive Recovery test to find switching threshold --- #
        # low-to-high transitions
        thres01, t_rgn01, t_rst01 = await self._get_overdrive_thres(tbm_specs, sim_dir, sim_db,
                                                                    dut, 1.0, '01')

        # high-to-low transitions
        thres10, t_rgn10, t_rst10 = await self._get_overdrive_thres(tbm_specs, sim_dir, sim_db,
                                                                    dut, -1.0, '10')

        mean_thres = (thres01 + thres10) / 2
        hysteresis = thres01 - mean_thres
        t_rst = max(t_rst01, t_rst10)
        t_rgn = max(t_rgn01, t_rgn10)

        output = dict(
            mean_thres=mean_thres,
            hysteresis=hysteresis,
            t_rgn=t_rgn,
            t_rst=t_rst,
        )

        if self.specs['estimate_tau']:
            # --- Estimate regeneration and resettling time constants by curve fitting --- #
            # low-to-high transitions
            tau_rgn01, tau_rst01 = await self._measure_taus(tbm_specs, sim_dir, sim_db, dut,
                                                            mean_thres, hysteresis, 1.0, '01')

            # high-to-low transitions
            tau_rgn10, tau_rst10 = await self._measure_taus(tbm_specs, sim_dir, sim_db, dut,
                                                            mean_thres, hysteresis, -1.0, '10')

            output['tau_rst'] = max(tau_rst01, tau_rst10)
            output['tau_rgn'] = max(tau_rgn01, tau_rgn10)

        return output

    async def _measure_taus(self, tbm_specs: Dict[str, Any], sim_dir: Path, sim_db: SimulationDB,
                            dut: DesignInstance, thres: float, hysteresis: float, sign: float,
                            suf: str) -> Tuple[float, float]:
        """Helper function to set up testbench managers for measuring time constants

        Parameters
        ----------
        tbm_specs : Dict[str, Any]
            Testbench manager specs
        sim_dir : Path
            Simulation directory
        sim_db : SimulationDB
            Simulation database
        dut : DesignInstance
            device under test
        thres : float
            switching threshold for this dut
        hysteresis : float
            hysteresis of switching threshold
        sign : float
            1.0 if differential input transitions from large negative to small positive
            -1.0 if differential input transitions from large positive to small negative
        suf : str
            '01' if sign is 1.0
            '10' if sign is -1.0

        Returns
        -------
        regeneration time constant, resettling time constant
        """
        specs = self.specs
        ov_wave = specs['ov_wave_params']
        cmode, ampl = ov_wave['cmode'], ov_wave['ampl']
        n_rst, n_eval = ov_wave['n_rst'], ov_wave['n_eval']

        # first eval amplitude
        tbm0_specs = deepcopy(tbm_specs)
        val0 = thres + sign * hysteresis + sign * 0.1e-3
        tbm0_specs['sup_values']['VDD_in'] = val0
        tbm0_specs['sup_values']['VSS_in'] = cmode - sign * ampl

        tbm0 = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm0_specs))
        tbm0.sim_params['t_sim'] = f't_rst + {n_rst + n_eval + 1}*tper'
        sim0_id = f'tau{suf}_{float_to_si_string(val0)}'
        sim0_results = await sim_db.async_simulate_tbm_obj(sim0_id, sim_dir / sim0_id, dut, tbm0,
                                                           {})

        # second eval amplitude
        tbm1_specs = deepcopy(tbm_specs)
        val1 = thres + sign * hysteresis + sign * 0.2e-3
        tbm1_specs['sup_values']['VDD_in'] = val1
        tbm1_specs['sup_values']['VSS_in'] = cmode - sign * ampl

        tbm1 = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm1_specs))
        tbm1.sim_params['t_sim'] = f't_rst + {n_rst + n_eval + 1}*tper'
        sim1_id = f'tau{suf}_{float_to_si_string(val1)}'
        sim1_results = await sim_db.async_simulate_tbm_obj(sim1_id, sim_dir / sim1_id, dut, tbm1,
                                                           {})

        tau_rgn, tau_rst = tbm1.get_taus(sim0_results.data, sim1_results.data, n_rst, sign,
                                         plot_flag=False)

        return tau_rgn, tau_rst

    async def _get_overdrive_thres(self, tbm_specs: Dict[str, Any], sim_dir: Path,
                                   sim_db: SimulationDB, dut: DesignInstance, sign: float, suf: str
                                   ) -> Tuple[float, float, float]:
        """Helper function to set up testbench managers for measuring switching threshold from
        overdrive recovery testbench

        Parameters
        ----------
        tbm_specs : Dict[str, Any]
            Testbench manager specs
        sim_dir : Path
            Simulation directory
        sim_db : SimulationDB
            Simulation database
        dut : DesignInstance
            device under test
        sign : float
            1.0 if differential input transitions from large negative to small positive
            -1.0 if differential input transitions from large positive to small negative
        suf : str
            '01' if sign is 1.0
            '10' if sign is -1.0

        Returns
        -------
        switching threshold, 10 - 90% regeneration time, 10 - 90% resettling time
        """
        specs = self.specs
        ov_wave = specs['ov_wave_params']
        cmode, ampl, tol = ov_wave['cmode'], ov_wave['ampl'], ov_wave['tol']
        n_rst, n_eval = ov_wave['n_rst'], ov_wave['n_eval']

        bin_iter = FloatBinaryIterator(cmode - ampl, cmode + ampl, tol)
        tbm = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm_specs))

        while bin_iter.has_next():
            val = bin_iter.get_next()
            tbm_specs['sup_values']['VDD_in'] = val
            tbm_specs['sup_values']['VSS_in'] = cmode - sign * ampl

            tbm = cast(CompTranTB, sim_db.make_tbm(CompTranTB, tbm_specs))
            tbm.sim_params['t_sim'] = f't_rst + {n_rst + n_eval + 1}*tper'
            sim_id = f'overdrive{suf}_{float_to_si_string(val)}'
            sim_results = await sim_db.async_simulate_tbm_obj(sim_id, sim_dir / sim_id, dut, tbm,
                                                              {})
            if tbm.check_overdrive_output(sim_results.data, sign, n_rst, n_eval):
                bin_iter.save_info(sim_results.data)
                bin_iter.down() if sign > 0 else bin_iter.up()
            else:
                bin_iter.up() if sign > 0 else bin_iter.down()

        thres = bin_iter.get_last_save()
        data = bin_iter.get_last_save_info()
        t_rgn, t_rst = tbm.get_times(data, n_rst)

        return thres, t_rgn, t_rst

    def _get_tbm_specs(self, env: str, vdd: float) -> Dict[str, Any]:
        specs = self.specs
        ov_wave = specs['ov_wave_params']
        cmode, ampl = ov_wave['cmode'], ov_wave['ampl']
        n_rst, n_eval = ov_wave['n_rst'], ov_wave['n_eval']
        const_params = specs['const_params']

        # create RST, RSTD, RSTB, v_REF, v_SIG signals
        pulse_list = [dict(pin='RST', tper='tper', tpw='tper/2', trf='trf'),
                      dict(pin='RSTD', tper='tper', tpw='tper/2', trf='trf', td='td'),
                      dict(pin='RSTB', tper='tper', tpw='tper/2', trf='trf', pos=False),
                      dict(pin='int_in', tper=f'{n_rst + n_eval}*tper', tpw=f'{n_eval}*tper',
                           trf='trf', td=f'{n_rst}*tper+trf')]
        pwr_domain = dict(RST=('VSS', 'VDD'),
                          RSTD=('VSS', 'VDD'),
                          RSTB=('VSS', 'VDD'),
                          int_in=('VSS_in', 'VDD_in'),
                          v_SIG=('VSS', 'VDD'),
                          v_REF=('VSS', 'VDD'),
                          v_BIAS=('VSS', 'VDD'),
                          v_OUTP=('VSS', 'VDD'),
                          v_OUTM=('VSS', 'VDD'),
                          )

        # add resistor and capacitors in TB
        load_list = [dict(pin='int_in', nin='v_SIG', type='res', value='rin'),
                     dict(pin='int_ref', nin='v_REF', type='res', value='rref'),
                     dict(pin='int_bias', nin='v_BIAS', type='res', value='rbias'),
                     dict(pin='v_SIG', type='cap', value='cin'),
                     dict(pin='v_REF', type='cap', value='cref'),
                     dict(pin='v_BIAS', type='cap', value='cbias'),
                     dict(pin='v_OUTP', type='cap', value='cload'),
                     dict(pin='v_OUTM', type='cap', value='cload'),
                     dict(pin='v_SAMP', nin='v_ACP', type='cap', value='cpre'),
                     dict(pin='v_COMPM', nin='v_ACM', type='cap', value='cpre'),
                     dict(pin='v_SWP', nin='v_COMPP', type='cap', value='cc'),
                     dict(pin='v_SWM', nin='v_COMPM', type='cap', value='cc')]

        tbm_specs = dict(
            dut_pins=['VDD', 'VSS', 'RST', 'RSTD', 'RSTB', 'RX_EN', 'v_BIAS', 'v_SIG',
                      'v_REF', 'v_COMPM', 'v_COMPP', 'v_OUTM', 'v_OUTP', 'v_SWM', 'v_SWP',
                      'v_ACM', 'v_ACP', 'v_SAMP'],
            pulse_list=pulse_list,
            pwr_domain=pwr_domain,
            load_list=load_list,
            sup_values=dict(VSS=0,
                            VDD=vdd,
                            RX_EN=vdd,
                            int_ref=cmode,
                            int_bias=const_params['vbias'],
                            ),
            sim_envs=[env],
            sim_params=dict(
                vdd=vdd,
                **specs['sim_params']
            ),
            sim_options=specs['sim_options'],
            tran_options=specs['tran_options'],
        )
        return tbm_specs
