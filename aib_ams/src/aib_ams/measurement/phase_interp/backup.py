from pprint import pprint
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from bag.io import open_file
from bag.simulation.core import MeasurementManager
from bag.simulation.data import SimData
from bag3_testbenches.measurement.data.tran import bits_to_pwl_iter
from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB


class PhaseInterpTimingTB(CombLogicTimingTB):
    """ Comb Logic Timing TB but with a second input inb that is offset from ina"""
    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ans = super(PhaseInterpTimingTB, self).pre_setup(sch_params)
        thres_lo: float = self.specs['thres_lo']
        thres_hi: float = self.specs['thres_hi']
        in_pwr: str = self.specs.get('stimuli_pwr', 'vdd')
        offset: str = self.specs['dly_offset']

        in_data = ['0', in_pwr, '0']
        trf_scale = f'{thres_hi - thres_lo:.4g}'
        inb_path = self.work_dir / 'inb_pwl.txt'
        with open_file(inb_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(in_data):
                f.write(f'{offset}+tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')

        in_file_list = ans['in_file_list']
        in_file_list.append(('inb', str(inb_path.resolve())))
        return ans


class PhaseInterpSeqTimingTB(CombLogicTimingTB):
    """ CombLogicSeqTimingTB but with a second input called inb """
    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        ans = super(PhaseInterpSeqTimingTB, self).pre_setup(sch_params)

        specs = self.specs
        thres_lo: float = specs['thres_lo']
        thres_hi: float = specs['thres_hi']
        in_pwr: str = specs.get('stimuli_pwr', 'vdd')
        offset: str = specs['dly_offset']

        _, num_runs = self._get_sim_time_info(specs)

        in_data = ['0', in_pwr, '0'] * num_runs
        trf_scale = f'{thres_hi-thres_lo:.4g}'
        in_path = self.work_dir / 'inb_pwl.txt'
        with open_file(in_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(in_data):
                f.write(f'{offset}+tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')
        in_file_list = ans['in_file_list']
        in_file_list.append(('inb', str(in_path.resolve())))
        return ans


class PhaseInterpMeasManager(MeasurementManager):

    def __init__(self, *args, **kwargs):
        super(PhaseInterpMeasManager, self).__init__(*args, **kwargs)
        nc = self.specs['num_codes']
        self.codes = self.specs.get('codes', [i for i in range(nc)])
        self.code_indx = 0
        self.tdrs = []
        self.tdfs = []
        self.trs = []
        self.tfs = []
        tb_params = self.get_default_tb_sch_params("delay")

        vbias_list = tb_params.get('vbias_list', [])
        vbias_list.extend(self._get_vbias_list())

        load_list = tb_params.get('load_list', [])
        load_list.append(['out', 'cload'])

        dut_conns = tb_params.get('dut_conns', dict())
        dut_conns.update(a_in='in', b_in='inb', VDD='VDD', VSS='VSS', out='out')
        dut_conns[f'a_en<{nc-1}:0>'] = ','.join(reversed([f'b{i}' for i in range(nc)]))
        dut_conns[f'a_enb<{nc-1}:0>'] = ','.join(reversed([f'b{i}b' for i in range(nc)]))
        dut_conns[f'b_en<{nc-1}:0>'] = ','.join(reversed([f'b{i}b' for i in range(nc)]))
        dut_conns[f'b_enb<{nc-1}:0>'] = ','.join(reversed([f'b{i}' for i in range(nc)]))

        tb_params.update(vbias_list=vbias_list, load_list=load_list, dut_conns=dut_conns)
        self.tb_params = tb_params

    def _get_vbias_list(self) -> List[List[str]]:
        nc: int = self.specs['num_codes']
        return [['VDD', 'vdd']] + [[f'b{i}b', f'vb{i}b'] for i in range(nc)] + \
               [[f'b{i}', f'vb{i}'] for i in range(nc)]

    def get_initial_state(self) -> str:
        return f'delay_0'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tb_name = self.get_testbench_name(state)
        tb_type = self.get_testbench_type(state)
        tb_specs = self.get_testbench_specs(tb_type)
        code = self.codes[self.code_indx]
        for i in range(self.specs['num_codes']):
            v = tb_specs['sim_params']['vdd']
            tb_specs['sim_params'][f'vb{i}'] = v if i >= code else 0
            tb_specs['sim_params'][f'vb{i}b'] = v if i < code else 0
        return tb_name, state, tb_specs, self.tb_params

    @staticmethod
    def get_testbench_type(state):
        if state.startswith('delay'):
            return 'delay'
        else:
            raise ValueError(f'Unknown state: {state}')

    def process_output(self,
                       state: str,
                       data: SimData,
                       tb_manager: CombLogicTimingTB,
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        tdr, tdf = tb_manager.get_output_delay(data, tb_manager.specs, 'in', 'out', False)
        tr, tf = tb_manager.get_output_trf(data, tb_manager.specs, 'XDUT.outb')
        self.trs.append(tr)
        self.tfs.append(tf)
        self.tdrs.append(tdr)
        self.tdfs.append(tdf)
        self.code_indx += 1
        if self.code_indx < len(self.codes):
            return False, f'delay_{self.codes[self.code_indx]}', dict()
        else:
            return True, '', dict(tdr=np.array(self.tdrs), tdf=np.array(self.tdfs),
                                  trs=np.array(self.trs), tfs=np.array(self.tfs))


class PhaseInterpSeqMeasManager(PhaseInterpMeasManager):
    def __init__(self, *args, **kwargs):
        super(PhaseInterpSeqMeasManager, self).__init__(*args, **kwargs)
        nc = self.specs['num_codes']
        # one time setup
        ctrl_params = dict()
        for i in range(nc):
            ctrl_params[f'b{i}'] = ['vdd'] * (i + 1) + ['0'] * (nc - i - 1)
            ctrl_params[f'b{i}b'] = ['0'] * (i + 1) + ['vdd'] * (nc - i - 1)
        self.ctrl_params = ctrl_params

    def _get_vbias_list(self) -> List[List[str]]:
        return [['VDD', 'vdd']]

    def get_initial_state(self) -> str:
        return f'delay'

    def get_testbench_info(self, state: str, prev_output: Optional[Dict[str, Any]]
                           ) -> Tuple[str, str, Dict[str, Any], Optional[Dict[str, Any]]]:
        tb_name = self.get_testbench_name(state)
        tb_type = self.get_testbench_type(state)
        tb_specs = self.get_testbench_specs(tb_type)
        tb_specs['ctrl_params'] = self.ctrl_params
        return tb_name, state, tb_specs, self.tb_params

    def process_output(self,
                       state: str,
                       data: SimData,
                       tb_manager: CombLogicTimingTB,
                       ) -> Tuple[bool, str, Dict[str, Any]]:
        tdr, tdf = tb_manager.get_output_delay(data, tb_manager.specs, 'in', 'out', False)
        tr, tf = tb_manager.get_output_trf(data, tb_manager.specs, 'XDUT.outb')
        return True, '', dict(tdr=tdr, tdf=tdf, trs=tr, tfs=tf)
