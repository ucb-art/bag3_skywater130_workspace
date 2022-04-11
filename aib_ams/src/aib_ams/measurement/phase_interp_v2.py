from typing import Dict, Any, Optional, Mapping

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB, _write_pwl_file


class PhaseInterpTimingTB(CombLogicTimingTB):
    """ Comb Logic Timing TB but with a second input inb that is offset from ina"""
    def pre_setup(self, sch_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        # ctrl_params
        self.specs['ctrl_params'] = ctrl_params = {}
        nc = self.specs['num_codes']
        for i in range(nc):
            ctrl_params[f'b{i}'] = ['vdd'] * (i + 1) + ['0'] * (nc - i - 1)
            ctrl_params[f'b{i}b'] = ['0'] * (i + 1) + ['vdd'] * (nc - i - 1)

        ans = super(PhaseInterpTimingTB, self).pre_setup(sch_params)

        # add extra input with offset
        thres_lo: float = self.specs['thres_lo']
        thres_hi: float = self.specs['thres_hi']
        in_pwr: str = self.specs.get('stimuli_pwr', 'vdd')
        offset: str = self.specs['dlycell_offset']
        write_numbers: bool = self.specs.get('write_numbers', False)
        swp_info = self.swp_info
        local_write_numbers = True
        if swp_info:
            if isinstance(swp_info, Mapping):
                if 'tbit' in swp_info or 'trf' in swp_info or in_pwr in swp_info:
                    local_write_numbers = False
            else:
                for l in swp_info:
                    if 'tbit' in l or 'trf' in l or in_pwr in l:
                        local_write_numbers = False

        write_numbers = write_numbers or local_write_numbers

        nbit_delay, num_runs = self._calc_sim_time_info()
        in_data = ['0', in_pwr, '0'] * num_runs
        trf_scale = thres_hi - thres_lo
        inb_path = self.work_dir / 'inb_pwl.txt'
        _write_pwl_file(inb_path, in_data, self.sim_params, 'tbit', 'trf', trf_scale, nbit_delay,
                        write_numbers, delay=offset)
        in_file_list = ans['in_file_list']
        in_file_list.append(('inb', str(inb_path.resolve())))
        return ans
