from typing import Optional, Dict, Any

from bag.io import open_file
from bag3_testbenches.measurement.data.tran import bits_to_pwl_iter
from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB


class StrongArmFlopTimingTB(CombLogicTimingTB):
    """ Strong Arm timing TB """
    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        specs = self.specs
        thres_lo: float = self.specs['thres_lo']
        thres_hi: float = self.specs['thres_hi']
        in_pwr: str = self.specs.get('stimuli_pwr', 'vdd')
        offset: str = self.specs['in_dly']
        rst_time: str = self.specs['rst_time']

        # generate PWL waveform files
        in_data = ['0', in_pwr, '0']
        inb_data = [in_pwr, '0', in_pwr]
        trf_scale = f'{thres_hi - thres_lo:.4g}'
        ina_path = self.work_dir / 'ina_pwl.txt'
        inb_path = self.work_dir / 'inb_pwl.txt'
        clk_path = self.work_dir / 'in_clk.txt'
        rstb_path = self.work_dir / 'rstb.txt'
        with open_file(ina_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(in_data):
                f.write(f'{rst_time}-{offset}+tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')
        with open_file(inb_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(inb_data):
                f.write(f'{rst_time}-{offset}+tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')
        with open_file(clk_path, 'w') as f:
            for _, s_tb, s_tr, val in bits_to_pwl_iter(in_data):
                f.write(f'{rst_time}+tbit*{s_tb}+trf*({s_tr})/{trf_scale} {val}\n')
        with open_file(rstb_path, 'w') as f:
            f.write(f'0 0\n')
            f.write(f'{rst_time} 0\n')
            f.write(f'{rst_time}+trf {in_pwr}\n')

        ans = sch_params.copy()
        ans['in_file_list'] = [('ina', str(ina_path.resolve())),
                               ('inb', str(inb_path.resolve())),
                               ('rstb', str(rstb_path.resolve()))]
        ans['clk_file_list'] = [('clk', str(clk_path.resolve()))]
        return ans
