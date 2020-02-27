from typing import cast, Dict, Any, Tuple, Union, Mapping, List

import numpy as np

from bag3_testbenches.measurement.tran.digital import DigitalTranTB, EdgeType
from bag.simulation.measure import MeasurementManager, MeasInfo
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimData
from bag3_liberty.data import parse_cdba_name

from bag.env import get_tech_global_info

from bag.simulation.cache import SimulationDB, DesignInstance, SimResults, MeasureResult


class DelaylineVerilogModelGenerator(MeasurementManager):

    def __init__(self, *args, **kwargs):
        # state interpretation guide:
        # sig_ctrlcode: update the delays due to signal change at ctrlcode
        # ctrl_{lo | hi}_ctrlbit: update the delays due to ctrl change at bit ctrlbit while
        # input is low | high
        super().__init__(*args, **kwargs)
        self._nand_delay: Dict[str, float] = {}

    def initialize(self, sim_db: SimulationDB, dut: DesignInstance) -> Tuple[bool, MeasInfo]:
        return False, MeasInfo('ctrl_lo_0', {})

    def process_output(self, cur_info: MeasInfo, sim_results: Union[SimResults, MeasureResult]
                       ) -> Tuple[bool, MeasInfo]:

        sim_params = self.specs['sim_params']
        sim_data: SimData = sim_results.data
        tbm: DigitalTranTB = cast(DigitalTranTB, sim_results.tbm)
        dut: DesignInstance = sim_results.dut
        tper = sim_params['tper']
        ncycles = sim_params['ncycles']
        ncodes = dut.lay_master.params['params']['num_insts']
        state, cur_code = self._get_code_state(cur_info)
        # signals = sim_data.signals
        # for sig in signals:
        #     if not sig.startswith('XDUT'):
        #         print(sig)

        cur_delays = {}
        done = (state == 'ctrl' and cur_code == ncodes)
        if state == 'sig':
            if cur_code < ncodes:
                srq = max(self._meas_srq_delay(cur_code, tbm, sim_data, tper, ncycles,
                                               input_fall=True),
                          self._meas_srq_delay(cur_code, tbm, sim_data, tper, ncycles,
                                               input_fall=False))
                nand_out = max(self._meas_nand_out_delay(cur_code, ncodes, tbm, sim_data, tper,
                                                         ncycles, input_fall=True),
                               self._meas_nand_out_delay(cur_code, ncodes, tbm, sim_data, tper,
                                                         ncycles, input_fall=False))
                cur_delays[f'srq_{cur_code}'] = srq
                cur_delays[f'nand_out_{cur_code}'] = nand_out

            if cur_code > 0:
                nand_in = max(self._meas_nand_in_delay(cur_code - 1, tbm, sim_data, tper,
                                                       ncycles, input_fall=True),
                              self._meas_nand_in_delay(cur_code - 1, tbm, sim_data, tper,
                                                       ncycles, input_fall=False))
                nand_out = max(self._meas_nand_out_delay(cur_code - 1, ncodes, tbm, sim_data, tper,
                                                         ncycles, input_fall=True),
                               self._meas_nand_out_delay(cur_code - 1, ncodes, tbm, sim_data, tper,
                                                         ncycles, input_fall=True))

                cur_delays[f'nand_in_{cur_code-1}'] = nand_in
                cur_delays[f'nand_out_{cur_code-1}'] = nand_out

            next_code = (cur_code + 1) % (ncodes + 1)
            next_state = 'ctrl_lo' if next_code == 0 else 'sig'
        elif state.startswith('ctrl'):
            input_val_str = state.split('_')[-1]
            input_val = 0 if input_val_str == 'lo' else 1

            nand_in = max(self._meas_nand_in_delay(cur_code, tbm, sim_data, tper, ncycles,
                                                   in0_to_output=False, input_fall=True),
                          self._meas_nand_in_delay(cur_code, tbm, sim_data, tper, ncycles,
                                                   in0_to_output=False, input_fall=False))

            nand_out = max(self._meas_nand_out_delay(cur_code, ncodes, tbm, sim_data, tper, ncycles,
                                                     in0_to_output=False, input_fall=True),
                           self._meas_nand_out_delay(cur_code, ncodes, tbm, sim_data, tper, ncycles,
                                                     in0_to_output=False, input_fall=False))

            srq_next = max(self._meas_srq_delay(cur_code + 1, tbm, sim_data, tper, ncycles,
                                                in0_to_output=True, input_fall=True),
                           self._meas_srq_delay(cur_code + 1, tbm, sim_data, tper, ncycles,
                                                in0_to_output=True, input_fall=False))

            srq_cur_r = self._meas_srq_delay(cur_code, tbm, sim_data, tper, ncycles,
                                             in0_to_output=False, input_fall=False)
            srqb_cur_r = self._meas_srqb_delay(cur_code, tbm, sim_data, tper, ncycles,
                                               in0_to_output=True, input_fall=False)
            srqb_cur_f = self._meas_srqb_delay(cur_code, tbm, sim_data, tper, ncycles,
                                               in0_to_output=True, input_fall=True)
            
            if (input_val == 0) == (cur_code % 2 == 1):
                if cur_code < ncodes - 1:
                    # till one before the last cell if input is lo and code is odd or if input is hi
                    # and code is even update srq(i+1)
                    # transition happens on both edges
                    cur_delays[f'srq_{cur_code + 1}'] = srq_next

                # if input == code.iseven(), update nand_in(i) for both transitions
                # for nand_out transition is only on input of nand_out rise
                cur_delays[f'nand_in_{cur_code}'] = nand_in
                cur_delays[f'nand_out_{cur_code}'] = nand_out
                cur_delays[f'srq_{cur_code}'] = srq_cur_r
                cur_delays[f'srqb_{cur_code}'] = srqb_cur_f
            else:
                cur_delays[f'srqb_{cur_code}'] = max(srqb_cur_r, srqb_cur_f)

            next_code = cur_code + 1 if input_val else cur_code
            next_state = 'ctrl_lo' if input_val else 'ctrl_hi'
        else:
            raise ValueError(f'{state} is not a valid state in this measurement manager.')

        self._update_delay_dict(cur_delays)

        print('-' * 30, f'code : {cur_code}', f'bk: {bin((1 << cur_code) - 1)}')
        print(self._nand_delay)
        next_info = f'{next_state}_{next_code}'
        print(f'cur_state: {cur_info.state}')
        print(f'next state: {next_info}')
        print('nand_out:', nand_out)
        print('nand_in:', nand_in)
        print('srq_next:', srq_next)
        print('srq_cur_r:', srq_cur_r)
        print('srqb_cur_r:', srqb_cur_r)
        print('srqb_cur_f:', srqb_cur_f)

        breakpoint()
        return done, MeasInfo(next_info, self._nand_delay)

    def get_sim_info(self, sim_db: SimulationDB, dut: DesignInstance, cur_info: MeasInfo
                     ) -> Tuple[Union[Tuple[TestbenchManager, Mapping[str, Any]],
                                      MeasurementManager], bool]:
        state, cur_code = self._get_code_state(cur_info)

        num_insts = dut.lay_master.params['params']['num_insts']
        sim_params = self.specs['sim_params']
        if state == 'sig':
            tbm_specs = self._get_tbm_params(cur_code, num_insts, sim_params)
        else:  # state == 'ctrl'
            input_val = 0 if state.split('_')[-1] == 'lo' else 1
            tbm_specs = self._get_ctrl_tbm_params(cur_code, num_insts, sim_params,
                                                  input_value=input_val)

        tbm = sim_db.make_tbm(DigitalTranTB, tbm_specs)
        tbm_info = tbm, {}
        return tbm_info, True

    def _update_delay_dict(self, update_dict: Dict[str, float]) -> None:
        """Update the delay dictionary by taking the max of the current content and the new value
        Parameters
        ----------
        update_dict: Dict[str, float]
            The update dictionary
        Returns
        -------
        None
        """
        for k, v in update_dict.items():
            cur_delay = self._nand_delay.get(k, float('-inf'))
            self._nand_delay[k] = max(cur_delay, v)

    def _get_code_state(self, cur_info: MeasInfo) -> Tuple[str, int]:
        split = cur_info.state.split('_')
        if len(split) == 2:
            return split[0], int(split[-1])
        elif len(split) == 3:
            return '_'.join(split[:2]), int(split[-1])
        else:
            raise ValueError(f'strange split {split}')

    def _meas_srq_delay(self, cur_code, tbm, sim_data, tper, ncycles, in0_to_output=True,
                        input_fall=True):
        in0_name = f'a<{cur_code - 1}>' if cur_code > 0 else 'dlyin'
        in1_name = f'srqb<{cur_code}>'
        out_name = f'srq<{cur_code}>'
        if in0_to_output:
            delay = self._meas_max_delay(tbm, sim_data, in0_name,
                                         out_name, tper, ncycles, input_fall)
        else:
            delay = self._meas_max_delay(tbm, sim_data, in1_name,
                                         out_name, tper, ncycles, input_fall)
        return delay

    def _meas_nand_in_delay(self, cur_code, tbm, sim_data, tper, ncycles, in0_to_output=True,
                            input_fall=True):
        in0_name = 'dlyin' if cur_code == 0 else f'a<{cur_code-1}>'
        in1_name = f'bk<{cur_code}>'
        out_name = f'a<{cur_code}>'
        if in0_to_output:
            delay = self._meas_max_delay(tbm, sim_data, in0_name,
                                         out_name, tper, ncycles, input_fall)
        else:
            delay = self._meas_max_delay(tbm, sim_data, in1_name,
                                         out_name, tper, ncycles, input_fall)
        return delay

    def _meas_nand_out_delay(self, cur_code, ncodes, tbm, sim_data, tper, ncycles,
                             in0_to_output=True, input_fall=True):
        in0_name = f'a<{cur_code-1}>' if cur_code == ncodes else f'b<{cur_code-1}>'
        in1_name = f'srq<{cur_code}>'

        out_name = 'dlyout' if cur_code == 0 else f'b<{cur_code-1}>'

        if in0_to_output:
            delay = self._meas_max_delay(tbm, sim_data, in0_name,
                                         out_name, tper, ncycles, input_fall)
        else:
            delay = self._meas_max_delay(tbm, sim_data, in1_name,
                                         out_name, tper, ncycles, input_fall)
        return delay

    def _meas_srqb_delay(self, cur_code, tbm, sim_data, tper, ncycles,
                         in0_to_output=True, input_fall=True):
        in0_name = f'bk<{cur_code}>'
        in1_name = f'srq<{cur_code}>'
        out_name = f'srqb<{cur_code}>'

        if in0_to_output:
            delay = self._meas_max_delay(tbm, sim_data, in0_name,
                                         out_name, tper, ncycles, input_fall)
        else:
            delay = self._meas_max_delay(tbm, sim_data, in1_name,
                                         out_name, tper, ncycles, input_fall)
        return delay

    def _meas_max_delay(self, tbm, sim_data, in_name, out_name, tper, ncycles,
                        input_fall=True):
        in_edge = EdgeType.FALL if input_fall else EdgeType.RISE
        out_edge = EdgeType.RISE if input_fall else EdgeType.FALL

        tdelay = tbm.calc_delay(sim_data, in_name, out_name,
                                in_edge=in_edge,
                                out_edge=out_edge,
                                t_start=tper * (ncycles - 1),
                                t_stop=tper * ncycles)
        return np.max(tdelay)

    def _get_ctrl_tbm_params(self,
                             code: int,
                             num_inst: int,
                             sim_params: Mapping[str, Any],
                             input_value: int,
                             ) -> Dict[str, Any]:

        global_info = get_tech_global_info('aib_ams')

        dut_pins = ['dlyin', 'CLKIN', 'VDD', 'VSS', 'iSE', 'RSTb',
                    'dlyout', 'iSI', 'SOOUT', f'a<{num_inst - 1}:0>',
                    f'b<{num_inst - 1}:0>', f'bk<{num_inst - 1}:0>', f'srq<{num_inst - 1}:0>',
                    f'srqb<{num_inst - 1}:0>']

        pwr_domain = {}
        for pin in dut_pins:
            pwr_domain[parse_cdba_name(pin)[0]] = ('VSS', 'VDD')

        pulse_list = [
            dict(
                pin=f'bk<{code}>',
                tper=sim_params['tper'],
                tpw=sim_params['tper'] / 2,
                trf=sim_params['trf'],
            ),
            dict(
                pin='CLKIN',
                tper=sim_params['tclk'],
                tpw=sim_params['tclk'] / 2,
                trf=sim_params['clk_trf'],
            )
        ]

        pin_values = {
                'dlyin': input_value,
                'iSE': 0,
                'iSI': 0,
                'RSTb': 1,
                f'a<{num_inst - 1}>': f'a<{num_inst - 1}>',
                f'b<{num_inst - 1}>': f'a<{num_inst - 1}>',
            }

        for i in range(code):
            pin_values[f'bk<{i}>'] = 1
        for i in range(code + 1, num_inst, 1):
            pin_values[f'bk<{i}>'] = 0

        default_sim_env = global_info['dsn_envs']['center']['env']
        sim_envs = sim_params.get('sim_envs', default_sim_env)
        tbm_specs = dict(
            sim_envs=sim_envs,
            sim_params=dict(
                t_sim=sim_params['ncycles'] * sim_params['tper'],
                t_rst=0,
                t_rst_rf=0,
            ),
            dut_pins=dut_pins,
            pulse_list=pulse_list,
            load_list=[dict(pin='dlyout', type='cap', value=sim_params['cload'])],
            pwr_domain=pwr_domain,
            sup_values=dict(VSS=0, VDD=global_info['vdd']),
            pin_values=pin_values,
            thres_lo=sim_params['thres_lo'],
            thres_hi=sim_params['thres_hi']
        )

        return tbm_specs

    def _get_tbm_params(self,
                        code: int,
                        num_inst: int,
                        sim_params: Mapping[str, Any],
                        ) -> Dict[str, Any]:
        global_info = get_tech_global_info('aib_ams')

        dut_pins = ['dlyin', 'CLKIN', 'VDD', 'VSS', 'iSE', 'RSTb',
                    'dlyout', 'iSI', 'SOOUT', f'a<{num_inst - 1}:0>',
                    f'b<{num_inst - 1}:0>', f'bk<{num_inst-1}:0>', f'srq<{num_inst - 1}:0>',
                    f'srqb<{num_inst - 1}:0>']

        pwr_domain = {}
        for pin in dut_pins:
            pwr_domain[parse_cdba_name(pin)[0]] = ('VSS', 'VDD')

        pulse_list = [
            dict(
                pin='dlyin',
                tper=sim_params['tper'],
                tpw=sim_params['tper'] / 2,
                trf=sim_params['trf'],
            ),
            dict(
                pin='CLKIN',
                tper=sim_params['tclk'],
                tpw=sim_params['tclk'] / 2,
                trf=sim_params['clk_trf'],
            )
        ]

        default_sim_env = global_info['dsn_envs']['center']['env']
        sim_envs = sim_params.get('sim_envs', default_sim_env)
        tbm_specs = dict(
            sim_envs=sim_envs,
            sim_params=dict(
                t_sim=sim_params['ncycles'] * sim_params['tper'],
                t_rst=0,
                t_rst_rf=0,
            ),
            dut_pins=dut_pins,
            pulse_list=pulse_list,
            load_list=[dict(pin='dlyout', type='cap', value=sim_params['cload'])],
            pwr_domain=pwr_domain,
            sup_values=dict(VSS=0, VDD=global_info['vdd']),
            pin_values={
                f'bk<{num_inst-1}:0>': (1 << code) - 1,
                'iSE': 0,
                'iSI': 0,
                'RSTb': 1,
                f'a<{num_inst-1}>': f'a<{num_inst-1}>',
                f'b<{num_inst-1}>': f'a<{num_inst-1}>',
                },
            thres_lo=sim_params['thres_lo'],
            thres_hi=sim_params['thres_hi']
        )

        return tbm_specs