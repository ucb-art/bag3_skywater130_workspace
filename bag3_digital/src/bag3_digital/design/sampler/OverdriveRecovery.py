# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Sequence, Optional, Union

from pathlib import Path

import numpy as np
import scipy.interpolate as interp
import scipy.optimize as sciopt
import matplotlib.pyplot as plt

from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData, AnalysisData, AnalysisType
)
from bag.simulation.base import SimAccess
from bag.simulation.core import MeasurementManager, TestbenchManager
from bag.simulation.hdf5 import save_sim_data_hdf5
from bag.math.interpolate import LinearInterpolator


class OverdriveTBM(TestbenchManager):

    def __init__(self, sim: SimAccess, work_dir: Path, tb_name: str, impl_lib: str,
                 specs: Dict[str, Any], sim_view_list: Sequence[Tuple[str, str]],
                 env_list: Sequence[str], precision: int = 6) -> None:
        TestbenchManager.__init__(self, sim, work_dir, tb_name, impl_lib, specs, sim_view_list,
                                  env_list, precision=precision)

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_setup = dict(
            sim_envs=self.sim_envs,
            analyses=[dict(type='TRAN',
                           start=0.0,
                           stop=self.specs['tsim'],
                           ),
                      ],
            params=dict(tper=self.specs['tper'],
                        tr=self.specs['tr'],
                        tdelay=self.specs['tdelay'],
                        tsim=self.specs['tsim'],
                        rst=self.specs['rst'],
                        vinit=self.specs['vinit'],
                        vfinal=self.specs['vfinal'],
                        vref=self.specs['vref'],
                        vincm=self.specs['vincm'],
                        vdd=self.specs['vdd'],
                        Vb=0.0,
                        vclk=self.specs['vclk'],
                        cload=self.specs['cload'],
                        cpre=self.specs['cpre'],
                        cc=self.specs['cc'],
                        rin=self.specs['rin'],
                        rref=self.specs['rref'],
                        ),
        )

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def compute_tset(cls, time: np.ndarray, sig: np.ndarray, plot_flag: bool = False) -> float:
        """
        This function computes time taken to settle from initial value of sig to final value of sig
        :param time: time axis
        :param sig: signal values corresponding to time axis
        :param plot_flag: True to plot sig vs time for debugging
        :return: time for 10 % - 90 % settling
        """
        if sig[0] > sig[-1]:
            sig = - sig
        err = np.abs(sig[-1] * 0.01)

        # find last index where sig is below final val - error
        idx = np.where(sig - (sig[-1] - err) < 0)[0][-1]
        fun = interp.InterpolatedUnivariateSpline(time, sig - (sig[-1] - err))
        tset = sciopt.brentq(fun, time[idx], time[idx + 1])

        # check for overshoot
        idx2_list = np.where(sig - (sig[-1] + err) > 0)[0]
        if idx2_list.size != 0:
            idx2 = idx2_list[-1]
            fun2 = interp.InterpolatedUnivariateSpline(time, sig - (sig[-1] + err))
            tset2 = sciopt.brentq(fun2, time[idx2], time[idx2 + 1])
            tset = max(tset, tset2)

        if plot_flag:
            print(tset)
            plt.figure()
            plt.plot(time, sig)
            plt.axvline(x=tset)
            plt.show()

        return tset

    def get_taus(self, data: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        This function computes tau_rst and tau_regen from the transient signal
        :param data: raw simulation data
        :return: tau_rst, tau_regen
        """
        tper = self.specs['tper']
        rst = self.specs['rst']
        tr = self.specs['tr']
        time = data['time']
        VoP, VoM = data['VoP'][0], data['VoM'][0]

        # compress time and sig arrays to one rise/fall edge, then call compute_tset
        # regen phase
        idx0 = np.where(time - (rst + 0.25) * tper - tr > 0)[0][0]
        idx1 = np.where(time - (rst + 0.75) * tper - tr > 0)[0][0]
        t_regen_P = self.compute_tset(time[idx0:idx1], VoP[idx0:idx1])
        t_regen_M = self.compute_tset(time[idx0:idx1], VoM[idx0:idx1])
        t_regen = max(t_regen_P, t_regen_M) - (rst + 0.5) * tper - tr

        # reset phase
        idx2 = np.where(time - (rst + 1.25) * tper - tr > 0)[0][0]
        t_rst_P = self.compute_tset(time[idx1:idx2], VoP[idx1:idx2])
        t_rst_M = self.compute_tset(time[idx1:idx2], VoM[idx1:idx2])
        t_rst = max(t_rst_P, t_rst_M) - (rst + 1) * tper - tr

        return t_rst / 4, t_regen / 4, VoP[idx1], VoM[idx1]

    @classmethod
    def plot_results(cls, results: Dict[str, Any]) -> None:
        time = results['time']
        Vin = results['Vin'][0]
        ViP, ViM = results['ViP'][0], results['ViM'][0]
        VoP, VoM = results['VoP'][0], results['VoM'][0]
        V1p, V1m = results['V1p'][0], results['V1m'][0]
        sw_p, sw_m = results['sw_p'][0], results['sw_m'][0]
        clk, clkd = results['clk'][0], results['clkd'][0]

        mode = 0

        if mode:
            plt.subplot(6, 1, 1)
            plt.plot(time, Vin)
            plt.ylabel('Input')

            plt.subplot(6, 1, 2)
            plt.plot(time, ViP)
            # plt.plot(time, ViM)
            plt.ylabel('Sampled Input')

            plt.subplot(6, 1, 3)
            plt.plot(time, V1p)
            # plt.plot(time, V1m)
            plt.ylabel('After AC cap')

            plt.subplot(6, 1, 4)
            plt.plot(time, clk)
            plt.plot(time, clkd)
            plt.ylabel('Clock')

            plt.subplot(6, 1, 5)
            plt.plot(time, VoP)
            plt.plot(time, VoM)
            plt.ylabel('Output')

            plt.subplot(6, 1, 6)
            plt.plot(time, sw_p)
            plt.plot(time, sw_m)
            plt.ylabel('Output before cross cap')
        else:
            plt.plot(time, clk)
            plt.plot(time, clkd)
            plt.plot(time, ViP)
            # plt.plot(time, ViM)
            plt.plot(time, V1p)
            # plt.plot(time, V1m)
            plt.plot(time, sw_p)
            plt.plot(time, sw_m)
            plt.plot(time, VoP)
            plt.plot(time, VoM)
            plt.legend(['clk', 'clkd', 'sampp', 'inp', 'gp', 'gn', 'outp', 'outn'])
            # plt.legend(['clk', 'clkd', 'sampp', 'sampn', 'inp', 'inn', 'gp', 'gn',
            #             'outp', 'outn'])

        plt.show()


class OverdriveMM(MeasurementManager):

    def __init__(self, sim: SimAccess, dir_path: Path, meas_name: str, impl_lib: str,
                 specs: Dict[str, Any], wrapper_lookup: Dict[str, str],
                 sim_view_list: Sequence[Tuple[str, str]], env_list: Sequence[str],
                 precision: int = 6) -> None:
        MeasurementManager.__init__(self, sim, dir_path, meas_name, impl_lib, specs,
                                    wrapper_lookup, sim_view_list, env_list, precision=precision)

    def get_initial_state(self):
        # type: () -> str
        """Returns the initial FSM state."""
        return 'overdrive'

    def process_output(self, state: str, data: Dict[str, Any], tb_manager: OverdriveTBM
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        done = True
        next_state = ''

        # tb_manager.plot_results(data['tran_meas-foreach'])
        tau_rst, tau_regen, VoP, VoM = tb_manager.get_taus(data['tran_meas-foreach'])

        output = dict(tau_rst=tau_rst,
                      tau_regen=tau_regen,
                      vop=VoP,
                      vom=VoM,
                      )

        return done, next_state, output
