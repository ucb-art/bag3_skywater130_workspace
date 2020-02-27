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

"""This package contains measurement class for PSS simulation of comparators."""

from typing import Optional, Tuple, Dict, Any, Union, Mapping, cast, List

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.data import (
    SimNetlistInfo, netlist_info_from_dict, SimData
)
from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.math.interpolate import LinearInterpolator
from bag.math import float_to_si_string
from bag.data.ltv import LTVImpulseFinite
from bag.util.immutable import Param
from bag.concurrent.util import GatherHelper

from ..tran.digital import DigitalTranTB


class CompPSSTB(DigitalTranTB):
    """This class sets up the comparator PSS measurement testbench.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._tsim = None
        super().__init__(*args, **kwargs)

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_params: Param = self.specs['sim_params']

        analyses_list = []
        tb_type = self.specs['tb_type']

        # PSS parameters
        tper = sim_params['tper']
        pss_dict = dict(type='PSS',
                        period=2 * tper,
                        options=dict(errpreset=self.specs['errpreset'],
                                     harms=self.specs['pss_harmonics'],
                                     tstab=50 * tper,
                                     skipdc='yes'))
        if 'PSS' in tb_type:
            pss_dict['options']['writepss'] = '"' + str(self.work_dir.parent / 'pss_state') + '"'
        else:
            try:
                del pss_dict['options']['writepss']
            except KeyError:
                pass
            pss_dict['options']['readpss'] = '"' + str(self.work_dir.parent / 'pss_state') + '"'
        analyses_list.append(pss_dict)

        if 'PAC' in tb_type:
            # PAC parameters
            pac_dict = dict(type='PAC',
                            sweep=dict(type='LINEAR',
                                       start=0,
                                       stop=self.specs['pac_sweep_stop'],
                                       num=self.specs['pac_sweep_num']),
                            options=dict(sweeptype='absolute',
                                         freqaxis='in',
                                         maxsideband=self.specs['pss_harmonics']),
                            save_outputs=self.specs['pac_save_outputs'])
            analyses_list.append(pac_dict)
        elif 'PNoise' in tb_type:
            # PNoise parameters
            freq_start = self.specs['pnoise_sweep_start']
            freq_stop = 1 / (2 * tper)
            pnoise_dict = dict(type='PNOISE',
                               sweep=dict(type='LOG',
                                          start=freq_start,
                                          stop=freq_stop,
                                          num=np.log(freq_stop / freq_start) * 10),
                               options=dict(sweeptype='absolute',
                                            maxsideband=self.specs['pss_harmonics'] // 2,
                                            refsideband=0,
                                            noisetype='timedomain',
                                            noisetimepoints=self.specs['noise_time_points'],
                                            numberofpoints=0),
                               p_port='v_COMPP',
                               n_port='v_COMPM',
                               in_probe='VPULSE3')  # TODO: hack
            analyses_list.append(pnoise_dict)

        sim_setup = self.get_netlist_info_dict()
        sim_setup.update(dict(analyses=analyses_list,
                              init_voltages=dict(v_OUTP=0.0,
                                                 v_OUTM=sim_params['vdd'])))

        return netlist_info_from_dict(sim_setup)

    @classmethod
    def plot_tran_waveforms(cls, sim_data: SimData) -> None:
        time = sim_data['time']
        Vin = sim_data['v_SIG'][0]
        Vref = sim_data['v_REF'][0]
        clk = sim_data['RSTB'][0]
        # clkd = sim_data['clkd'][0]
        VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]

        VsampP = sim_data['v_SAMP'][0]
        VacP = sim_data['v_ACP'][0]
        VintM = sim_data['v_SWM'][0]
        VcompP, VcompM = sim_data['v_COMPP'][0], sim_data['v_COMPM'][0]

        mode = 0

        if mode:
            plt.subplot(5, 1, 1)
            plt.plot(time, clk)
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
            plt.plot(time, clk, label='clk')
            # plt.plot(time, clkd, label='clkd')
            plt.plot(time, Vin, label='inp')
            # plt.plot(time, Vref)
            plt.plot(time, VsampP, label='sampp')
            plt.plot(time, VacP, label='acp')
            plt.plot(time, VcompP, label='compp')
            plt.plot(time, VcompM, label='compm')
            # plt.plot(time, VintP)
            # plt.plot(time, VintM)
            plt.plot(time, VoP, label='outp')
            plt.plot(time, VoM, label='outm')
            plt.legend()
            plt.xlabel('time (in sec)')
            plt.ylabel('Ampl (in V)')

        plt.show()

    @classmethod
    def plot_ac_waveforms(cls, sim_data: SimData) -> None:
        # TODO: first corner, third rising edge (based on pwl wave), fundamental frequency
        freq = sim_data['freq'][0][0][7]
        Vin = sim_data['v_SIG'][0][0][7]
        # Vref = sim_data['v_REF'][0]
        # clk = sim_data['clk'][0]
        # clkd = sim_data['clkd'][0]
        # VoP, VoM = sim_data['v_OUTP'][0], sim_data['v_OUTM'][0]

        VsampP = sim_data['v_SAMP'][0][0][7]
        VacP = sim_data['v_ACP'][0][0][7]
        VcompP, VcompM = sim_data['v_COMPP'][0][0][7], sim_data['v_COMPM'][0][0][7]

        mode = 0

        if mode:
            plt.subplot(5, 1, 1)
            # plt.plot(freq, clk)
            # plt.plot(freq, clkd)
            plt.ylabel('Clock')

            plt.subplot(5, 1, 2)
            # plt.plot(freq, Vin)
            # plt.plot(freq, Vref)
            plt.ylabel('Input')

            plt.subplot(5, 1, 3)
            # plt.plot(freq, VsampP)
            plt.plot(freq, abs(VacP))
            plt.ylabel('Sampled, AC Cap')

            plt.subplot(5, 1, 4)
            # plt.plot(freq, VcompP)
            plt.plot(freq, abs(VcompM))
            plt.ylabel('Comp Output')

            plt.subplot(5, 1, 5)
            # plt.plot(freq, VoP)
            # plt.plot(freq, VoM)
            plt.ylabel('Final Output')

        else:
            # plt.plot(freq, clk)
            # plt.plot(freq, clkd)
            plt.semilogx(freq, abs(Vin))
            # plt.plot(freq, Vref)
            plt.semilogx(freq, abs(VsampP))
            plt.semilogx(freq, abs(VacP))
            # plt.plot(freq, VcompP)
            plt.semilogx(freq, abs(VcompM))
            # plt.plot(freq, abs(VcompM / VacP))
            # plt.plot(freq, VintP)
            # plt.plot(freq, VintM)
            # plt.plot(freq, VoP)
            # plt.plot(freq, VoM)
            # plt.plot(freq, VDD_left)
            # plt.plot(freq, VDD_mid)
            # plt.legend(['clk', 'clkd', 'inp', 'sampp', 'acp', 'compp', 'compm', 'intm', 'outp',
            #             'outm'])
            plt.legend(['in', 'sampp', 'acp', 'compm'])
            plt.xlabel('Frequency (in Hz)')
            plt.ylabel('Magnitude')

        plt.show()

    def post_process_pac(self, sim_data: SimData, debug: bool = False, plot_flag: bool = False
                         ) -> Dict[str, Any]:
        """NOTE: PSS arbitrarily cyclic shifts time time domain signals, but PAC computes
        everything wrt the original transient waveform. So all PSS time domain waveforms have
        to be cyclic shifted, but PAC waveforms do not need any phase shift.

        Also, PSS time axis is not equispaced, so resample all time domain signals before cyclic
        shifting to avoid glitches.

        Parameters
        ----------
        sim_data : SimData
            the simulation data
        debug : bool
            True to debug using 'lsim' of LTVImpulseFinite
        plot_flag : bool
            True to plot waveforms for debugging

        Returns
        -------
        Dictionary of processed PAC results
        """

        sim_params: Param = self.specs['sim_params']
        tper = sim_params['tper']
        trf = sim_params['trf']
        t_rst = sim_params['t_rst']
        t_rst_rf = sim_params['t_rst_rf']

        # PAC frequency domain waveforms
        sim_data.open_group('pac')
        m = (sim_data['harmonic'].size - 1) // 2
        freq = sim_data['freq']
        n = freq.size - 1
        v_comp_pac = sim_data['v_COMPP'][0] - sim_data['v_COMPM'][0]  # first corner
        fstep = freq[-1] // n

        # PSS time domain waveforms
        sim_data.open_group('pss_td')
        time = sim_data['time']
        t2 = np.linspace(start=0, stop=time[-1], num=1000, endpoint=False)
        # resample everything with t2 because PSS is weird and time axis is not equally spaced
        v_in_td_orig = LinearInterpolator([time], sim_data['int_in'][0], [1e-12])(t2)
        pss_tper = 2 * tper
        if np.abs(time[-1] - 2 * tper) > tper / 1000:
            raise ValueError(f'Rest of post processing assumes that PSS is run for {2 * tper} sec')

        k = int(1 / (fstep * pss_tper))

        # Figure out cyclic shifts
        true_rise = t_rst + t_rst_rf / 0.80 + tper + trf  # for 2 period input
        t2_idx = np.where(t2 >= true_rise)[0][0]  # index where everything has to be shifted

        v_in_td_diff = v_in_td_orig - (np.max(v_in_td_orig) + np.min(v_in_td_orig)) / 2
        PAC_idx = np.where(np.diff((v_in_td_diff > 0).astype(int)) > 0)[0][0]
        roll = t2_idx - PAC_idx

        # Do cyclic shifts after resampling
        def _resample_and_roll(in_arr: np.ndarray) -> np.ndarray:
            int_arr = LinearInterpolator([time], in_arr, [1e-12])(t2)
            return np.roll(int_arr, roll)

        vdd = _resample_and_roll(sim_data['VDD'][0])
        v_in_td = np.roll(v_in_td_orig, roll)
        rst = _resample_and_roll(sim_data['RST'][0])
        clk = vdd - rst
        rstd = _resample_and_roll(sim_data['RSTD'][0])
        clkd = vdd - rstd
        v_compP_td = _resample_and_roll(sim_data['v_COMPP'][0])  # first corner
        v_compM_td = _resample_and_roll(sim_data['v_COMPM'][0])  # first corner
        v_comp_td = v_compP_td - v_compM_td
        v_swP_td = _resample_and_roll(sim_data['v_SWP'][0])  # first corner
        v_swM_td = _resample_and_roll(sim_data['v_SWM'][0])  # first corner
        v_acP_td = _resample_and_roll(sim_data['v_ACP'][0])  # first corner
        v_acM_td = _resample_and_roll(sim_data['v_ACM'][0])  # first corner
        v_sampP_td = _resample_and_roll(sim_data['v_SAMP'][0])  # first corner
        # ibias_td = _resample_and_roll(sim_data['Xiprb:in'][0])  # first corner

        out0 = np.vstack((t2, v_comp_td)).T

        # create 2D impulse response
        ltv = LTVImpulseFinite(v_comp_pac, m, n, pss_tper, k, out0)
        tau = np.linspace(start=tper, stop=2*tper, num=100, endpoint=False)
        t_gr, tau_gr = np.meshgrid(t2, tau, indexing='ij')
        val = ltv(t_gr, tau_gr)

        # integrate across tau to get gain as a function of output time t2
        gain: np.ndarray = np.trapz(val, tau, axis=1)

        # DEBUG
        # debug_dict = {}
        # if debug:
        #     # lsim
        #     input_wave = np.zeros_like(time)
        #     time_idx_ini = np.where(time >= max_gain2_tau)[0][0]
        #     time_idx_fin = np.where(time >= max_gain2_tau + 10.0e-12)[0][0]
        #     input_wave[time_idx_ini:time_idx_fin] += 1.0e-6
        #     output_wave = ltv.lsim(input_wave, time[1] - time[0])
        #
        #     impulse_delay = max_gain2_tau - true_rise
        #     debug_dict = dict(impulse_response=impulse_response,
        #                       impulse_delay=impulse_delay,
        #                       input_wave=input_wave,
        #                       output_wave=output_wave,)

        # find max gain index
        max_gain_idx = np.argmax(np.abs(gain))

        num_idx = 30
        margin = 10.0e-12

        # find second rise edge of clkd
        clkd_diff = clkd - (np.max(clkd) + np.min(clkd)) / 2
        rise_idx = np.where(np.diff((clkd_diff > 0).astype(int)) > 0)[0][1]
        time_idx2 = np.where(t2 >= t2[rise_idx] + trf / 2)[0][0]

        ini_idx = np.where(t2 >= t2[time_idx2] - margin / 2)[0][0]
        fin_idx = np.where(t2 >= t2[max_gain_idx] + margin / 2)[0][0]
        time_indices = np.linspace(ini_idx, fin_idx, num_idx, dtype=int)

        time_max_gain = t2[time_indices] - true_rise
        PAC_gain = gain[time_indices]

        if plot_flag:
            plt.subplot(211)
            plt.plot(t2, clk, label='clk')
            plt.plot(t2, clkd, label='clkd')
            plt.plot(t2, v_compP_td, label='compP')
            plt.plot(t2, v_compM_td, label='compM')
            plt.plot(t2, v_comp_td, label='comp')
            plt.plot(t2, v_swP_td, label='sw_P')
            plt.plot(t2, v_swM_td, label='sw_M')
            plt.plot(t2, v_in_td, label='in')
            for axv in time_max_gain:
                plt.axvline(x=axv + true_rise)
            plt.legend()

            plt.subplot(212)
            plt.plot(t2, gain, label='small_sig')
            plt.legend()

            # plt.subplot(313)
            # plt.plot(t2, ibias_td, label='ibias')
            # # plt.plot(time, noise_PNoise, label='gain')
            # plt.xlabel('time (in sec)')
            # plt.ylabel('Current (A)')
            # plt.legend()
            plt.show()

        sig_dict = dict(
            clk=clk,
            clkd=clkd,
            compP=v_compP_td,
            compM=v_compM_td,
            comp=v_comp_td,
            swP=v_swP_td,
            swM=v_swM_td,
            acP=v_acP_td,
            acM=v_acM_td,
            sampP=v_sampP_td,
            # ibias_tot=ibias_td,
        )

        return dict(sig_dict=sig_dict,
                    gain=gain,
                    time=t2,
                    PAC_idx=t2_idx,
                    time_max_gain=time_max_gain,
                    true_rise=true_rise,
                    PAC_gain=PAC_gain,
                    # **debug_dict,
                    )

    @classmethod
    def post_process_pnoise(cls, sim_data: SimData, pac_dict: Dict[str, Any]) -> Dict[str, Any]:
        """NOTE: PSS arbitrarily cyclic shifts time time domain signals, PNoise waveforms are
        shifted to follow the PSS. So all PSS and PNoise time domain waveforms have to be cyclic
        shifted to original transient waveforms for further post processing and comparison with
        PAC, etc.

        Parameters
        ----------
        sim_data : SimData
            the simulation data
        pac_dict : Post processed PAC results

        Returns
        -------
        Dictionary of processed PNoise results
        """

        # PNoise frequency domain waveforms
        sim_data.open_group('pnoise')
        noise_out = sim_data['out'][0]
        freq = sim_data['freq']
        time_index = sim_data['timeindex']

        # Final output noise: calculate integrated noise at different time points
        integ_out_noise_arr = np.zeros(time_index.size)
        freq_log = np.log(freq)
        for idx in range(time_index.size):
            noise_fun = LinearInterpolator([freq_log], np.log(noise_out[idx] ** 2), [1e-9],
                                           extrapolate=True)
            integ_out_noise_arr[idx] = np.sqrt(noise_fun.integrate(freq_log[0], freq_log[-1],
                                                                   logx=True, logy=True, raw=True))

        # sim_data.open_group('pss_td')
        # clk = sim_data['clk'][0]
        # time = sim_data['time']
        #
        # time_PAC = pac_dict['time']
        # clk = LinearInterpolator([time], clk, [1e-12])(time_PAC)
        #
        # clk_diff = (np.append(np.diff(clk), clk[0] - clk[-1]) > 0).astype(int)
        # PNoise_idx = np.argmin(abs(clk_diff * clk - 0.5))
        # PAC_idx = pac_dict['PAC_idx']
        # roll = PAC_idx - PNoise_idx
        # clk_roll = np.roll(clk, roll)
        #
        # integ_out_noise_arr = np.append(integ_out_noise_arr, integ_out_noise_arr[0])
        # time_index = np.append(time_index, time[-1])
        # noise = LinearInterpolator([time_index], integ_out_noise_arr, [1e-12])(time_PAC)
        # noise_roll = np.roll(noise, roll)
        #
        # max_gain_idx = (pac_dict['max_gain_idx'] - roll) % time_PAC.size
        # noise_idx = np.argmin(np.abs(time_index - time_PAC[max_gain_idx]))

        # at time_index, integrate all noise signals
        integ_noise_dict = {}
        for tidx in range(time_index.size):
            components = {}
            for sig in sim_data.signals:
                if not (sig.endswith('total') or sig == 'out' or 0.0 in sim_data[sig][0][tidx]):
                    noise_fun = LinearInterpolator([freq_log], np.log(sim_data[sig][0][tidx]),
                                                   [1e-9], extrapolate=True)
                    components[sig] = np.sqrt(noise_fun.integrate(freq_log[0], freq_log[-1],
                                                                  logx=True, logy=True, raw=True))
            integ_noise_dict[tidx] = components

        # SANITY CHECK
        # out_pdf = noise_out[noise_idx]
        # calc_pdf = np.zeros(out_pdf.size)
        # for sig in sim_data.signals:
        #     if not (sig.endswith('total') or sig == 'out'):
        #         calc_pdf += sim_data[sig][0][noise_idx]
        # calc_pdf = np.sqrt(calc_pdf)

        return dict(
            noise_out=integ_out_noise_arr,
            # noise_out=integ_out_noise_arr[pac_dict['max_gain_idx']],
            noise_components=integ_noise_dict,
            # clk=clk_roll,
            # noise=noise_roll,
        )
    
    @classmethod
    def final_process_pss(cls, output_pac: Dict[str, Any], output_pnoise: Dict[str, Any],
                          plot_flag: bool = False) -> float:
        """Process PAC and PNoise results to estimate input referred noise

        Parameters
        ----------
        output_pac : Dict[str, Any]
            post processed PAC results
        output_pnoise : Dict[str, Any]
            post processed PNoise results
        plot_flag : bool
            True to plot waveforms for debugging

        Returns
        -------
        Standard deviation of input referred noise (in V)
        """
        # assume that PAC time is true time
        # PAC data
        time = output_pac['time']

        sig_dict = output_pac['sig_dict']
        clk_PAC = sig_dict['clk']
        clkd = sig_dict['clkd']
        compP = sig_dict['compP']
        compM = sig_dict['compM']
        comp = sig_dict['comp']
        swP = sig_dict['swP']
        swM = sig_dict['swM']
        acP = sig_dict['acP']
        acM = sig_dict['acM']
        sampP = sig_dict['sampP']
        # ibias_tot = sig_dict['ibias_tot']

        gain = output_pac['gain']
        PAC_gain = output_pac['PAC_gain']

        # PNoise data
        noise = output_pnoise['noise_out']
        # clk_PNoise = output_pnoise['clk']
        # noise_PNoise = output_pnoise['noise']

        # find max gain point
        # gain_idx = np.argmax(np.abs(gain))
        out_noise_sq = noise ** 2
        # find input referred noise
        # input_noise = np.abs(noise / gain[gain_idx])
        input_noise = np.abs(noise / PAC_gain)
        print(f'Output referred noise (in V) = {noise}')
        print(f'Input referred noise (in V) = {input_noise}')
        print(f'PAC gain (in V/V) = {PAC_gain}')

        noise_components: Dict[int, Dict[str, float]] = output_pnoise['noise_components']
        for tidx in range(len(noise_components)):
            components: Dict[str, float] = noise_components[tidx]
            sorted_components: List[Tuple[str, float]] = sorted(components.items(),
                                                                key=lambda kv: kv[1], reverse=True)
            print(f'Top noise components at time index {tidx} are:')
            for i in sorted_components[:5]:
                print(f'{i[0]} : {i[1]**2 / out_noise_sq[tidx] * 100}%')

        if plot_flag:
            plt.figure()
            plt.subplot(211)
            plt.plot(time, clk_PAC, label='clk')
            plt.plot(time, clkd, label='clkd')
            plt.plot(time, compP, label='compP')
            plt.plot(time, compM, label='compM')
            plt.plot(time, comp, label='comp')
            # plt.plot(time, swP, label='swP')
            # plt.plot(time, swM, label='swM')
            # plt.plot(time, acP, label='acP')
            # plt.plot(time, acM, label='acM')
            plt.plot(time, sampP, label='sampP')
            # plt.plot(time, out_en, label='out_en')
            for axv in output_pac['time_max_gain']:
                plt.axvline(x=axv + output_pac['true_rise'])
            plt.xlabel('time (in sec)')
            plt.ylabel('Voltage (in V)')
            plt.legend()

            plt.subplot(212)
            plt.plot(time, gain, label='small_sig')
            plt.xlabel('time (in sec)')
            plt.ylabel('Absolute (V/V)')
            # plt.ylabel('Percentage')
            plt.legend()

            # plt.subplot(313)
            # plt.plot(time, ibias_tot, label='total')
            # plt.xlabel('time (in sec)')
            # plt.ylabel('Current (A)')
            # plt.legend()

            plt.show()

        return input_noise


class CompPSSMM(MeasurementManager):
    """This class performs measurements based on PSS, PAC and PNoise simulations"""

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
        results = self.specs['tran_results']

        idx = 0
        keys = ['noise_in']

        for env in corners['envs']:
            vdd_list = corners['vdd'][env]
            results[env].update({key: [] for key in keys})
            for vdd in vdd_list:
                for key in keys:
                    results[env][key].append(coro_results[idx][key])
                idx += 1

        return results

    async def meas_cor_vdd(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                           dut: Optional[DesignInstance], env: str, vdd: float) -> Dict[str, Any]:
        """Run measurement per corner and supply voltage"""
        sim_dir = sim_dir / env / float_to_si_string(vdd)

        # 0. do PSS + PAC measurements
        tbm_specs = self._get_tbm_specs(env, vdd, 'PAC')
        tbm_specs['tb_type'] = sim0_id = 'PSS_PAC'
        tbm0 = cast(CompPSSTB, sim_db.make_tbm(CompPSSTB, tbm_specs))
        sim0_results = await sim_db.async_simulate_tbm_obj(sim0_id, sim_dir / sim0_id, dut, tbm0,
                                                           {})
        output_pac = tbm0.post_process_pac(sim0_results.data, plot_flag=False)

        # 1. do PSS to find time points for PNoise
        tbm_specs = self._get_tbm_specs(env, vdd, 'PNoise')
        tbm_specs['tb_type'] = sim1_id = 'PSS'
        tbm1 = cast(CompPSSTB, sim_db.make_tbm(CompPSSTB, tbm_specs))
        sim1_results = await sim_db.async_simulate_tbm_obj(sim1_id, sim_dir / sim1_id, dut, tbm1,
                                                           {})
        sim1_data = sim1_results.data
        sim1_data.open_group('pss_td')
        time = sim1_data['time']
        t2 = np.linspace(start=0, stop=time[-1], num=1000, endpoint=False)
        in_td = LinearInterpolator([time], sim1_data['int_in'][0], [1e-12])(t2)
        in_td_diff = in_td - (np.max(in_td) + np.min(in_td)) / 2
        in_idx = np.where(np.diff((in_td_diff > 0).astype(int)) > 0)[0][0]
        rise_edge = t2[in_idx]
        tbm_specs['noise_time_points'] = list((rise_edge + output_pac['time_max_gain']) % time[-1])

        # 2. do PNoise
        tbm_specs['tb_type'] = sim2_id = 'PNoise'
        tbm2 = cast(CompPSSTB, sim_db.make_tbm(CompPSSTB, tbm_specs))
        sim2_results = await sim_db.async_simulate_tbm_obj(sim2_id, sim_dir / sim2_id, dut, tbm2,
                                                           {})
        output_pnoise = tbm2.post_process_pnoise(sim2_results.data, output_pac)

        # 3. calculate input referred noise
        noise_in = tbm2.final_process_pss(output_pac, output_pnoise, plot_flag=True)

        return dict(noise_in=list(noise_in))

    def _get_tbm_specs(self, env: str, vdd: float, mode: str) -> Dict[str, Any]:
        specs = self.specs
        const_params = specs['const_params']
        thres_dev = const_params['thres_dev']

        tran_results = specs['tran_results']
        idx = tran_results[env]['vdd'].index(vdd)
        thres = tran_results[env]['mean_thres'][idx]
        hysteresis = tran_results[env]['hysteresis'][idx]

        # create RST, RSTD, RSTB, v_REF, v_SIG signals
        pulse_list = [dict(pin='RST', tper='tper', tpw='tper/2', trf='trf'),
                      dict(pin='RSTD', tper='tper', tpw='tper/2', trf='trf', td='td'),
                      dict(pin='RSTB', tper='tper', tpw='tper/2', trf='trf', pos=False),
                      dict(pin='int_in', tper='2*tper', tpw='tper', trf='trf', td='tper+trf',
                           extra=dict(pacmag=1))]
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
                            VSS_in=const_params['vampl_ini'],
                            VDD_in=thres + hysteresis + thres_dev,
                            RX_EN=vdd,
                            int_ref=const_params['vref'],
                            int_bias=const_params['vbias'],
                            ),
            sim_envs=[env],
            sim_params=dict(
                vdd=vdd,
                t_sim='t_rst + 4*tper',
                **specs['sim_params']
            ),
            sim_options=specs['sim_options'],
            **self.specs[mode]
        )

        return tbm_specs
