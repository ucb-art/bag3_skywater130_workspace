from typing import Dict, Any, Tuple, Optional, Iterable, cast

from pathlib import Path
import pprint
import pdb
from math import ceil, floor, sqrt, log
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from scipy.optimize import brentq
import argparse

from bag.simulation.core import MeasurementManager, DesignManager
from bag.simulation.data import SimData
from bag.layout.template import TemplateDB
from bag.io.file import read_yaml
from bag.core import BagProject
from bag.util.immutable import to_immutable, ImmutableType
from bag.design.database import ModuleDB
from bag.util.search import BinaryIterator

from bag3_testbenches.measurement.digital.db import DigitalDB
from bag3_testbenches.measurement.digital.enum import TimingMeasType
from bag3_testbenches.measurement.digital.timing import OutputDriverTB
from bag3_testbenches.measurement.digital.exception import TimeTooSmallError


class OutputDriverMeasManager(MeasurementManager):

    def __init__(self, *args, **kwargs):
        super(OutputDriverMeasManager, self).__init__(*args, **kwargs)

    def get_initial_state(self) -> str:
        tbs = self.specs['testbenches']
        if len(tbs) > 1:
            raise ValueError('OutputDriverMeasManager cannot have more than one tb')
        return list(tbs.keys())[0]

    def process_output(self,
                       state: str,
                       data: SimData,
                       tb_manager: OutputDriverTB,
                       ) -> Tuple[bool, str, Dict[str, Any]]:

        tper = tb_manager.specs['sim_params']['tper']
        cout = tb_manager.specs['sim_params']['cout']
        vdd = tb_manager.specs['sim_params']['vdd']
        if state == 'output_res':
            res_data = tb_manager.get_output_res(data, tper, cout, vdd=vdd)

            results = dict(
                pullup_res=res_data['pu'],
                pulldown_res=res_data['pd'],
            )

        elif state == 'tr_tf':
            trf_data = tb_manager.get_tr_tf(data, tper, vdd_output=vdd)
            # trf_data = tb_manager.get_tr_tf(data, tper, clk_trf, vdd=vdd, out_key='midp')
            results = dict(
                tr=trf_data['tr'],
                tf=trf_data['tf'],
            )
            print(f'tr: {results["tr"]}')
            print(f'tf: {results["tf"]}')

        elif state == 'tdelay':
            vdd_input = tb_manager.specs['sim_params']['vdd_core']
            td_in_midp = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                              in_key='in',  out_key='midp')
            td_midp_midn = tb_manager.get_delay(data, tper, vdd_input=vdd, vdd_output=vdd,
                                                in_key='midp', out_key='midn')
            td_in_midn = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                              in_key='in', out_key='midn')
            td_in_inb = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd_input,
                                             in_key='in', out_key='inbbuf')
            td_inb_midn = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
                                               in_key='inbbuf', out_key='midn')
            # td_in_midn2 = tb_manager.get_delay(data, tper, vdd_input=vdd_input, vdd_output=vdd,
            #                                    in_key='in', out_key='midn')

            # print('l2h in -> midn (1)')
            # print(f'in -> midp: {td_in_midp["tdl2h"]:6.4e}, '
            #       f'midp -> midn: {td_midp_midn["tdh2l"]:6.4g}')
            # print(f'in -> midn (1): {td_in_midn["tdl2h"]:6.4g}')
            #
            # print('h2l in -> midn (1)')
            # print(f'in -> midp: {td_in_midp["tdh2l"]:6.4e}, '
            #       f'midp -> midn: {td_midp_midn["tdl2h"]:6.4g}')
            # print(f'in -> midn (1): {td_in_midn["tdh2l"]:6.4g}')
            #
            # print('l2h in -> midn (2)')
            # print(f'in -> inb: {td_in_inb["tdl2h"]:6.4e}, '
            #       f'inb -> midn: {td_inb_midn["tdh2l"]:6.4g}')
            # print(f'in -> midn (2): {td_in_midn2["tdl2h"]:6.4g}')
            #
            # print('h2l in -> midn (2)')
            # print(f'in -> inb: {td_in_inb["tdh2l"]:6.4e}, '
            #       f'inb -> midn: {td_inb_midn["tdl2h"]:6.4g}')
            # print(f'in -> midn (2): {td_in_midn2["tdh2l"]:6.4g}')

            plt.clf()
            plt.subplot(211)
            # plt.plot(data['time'], data['inbuf'][0], label='inbuf')
            plt.plot(data['time'], data['inbbuf'][0], label='inbbuf')
            plt.plot(data['time'], data['in'][0], label='in')
            plt.legend()
            plt.subplot(212)
            plt.plot(data['time'], data['inbbuf'][0], label='inbbuf')
            plt.plot(data['time'], data['in'][0], label='in')
            plt.plot(data['time'], data['midn'][0], label='midn')
            plt.plot(data['time'], data['midp'][0], label='midp')
            plt.legend()

            results = dict(
                ttop1=td_in_midp["tdl2h"],
                ttop2=td_midp_midn["tdh2l"],
                tbot1=td_in_inb["tdh2l"],
                tbot2=td_inb_midn["tdl2h"],
                ttop=td_in_midn["tdl2h"],
                tbot=td_in_midn["tdh2l"],
            )

            if self.specs['plot_figs']:
                pprint.pprint(results)
                plt.show()

        else:
            raise KeyError('Invalid state!')

        return True, '', results


class DesignDB:

    def __init__(self, groups: Iterable[str]):
        self._design_db = {}
        for kwrd in groups:
            self._design_db[kwrd] = {}

        self._cur_group = next(iter(self._design_db.values()))
        self._cur_group_name = next(iter(self._design_db.keys()))

    @property
    def group_list(self):
        return list(self._design_db.keys())

    @property
    def group(self):
        return self._cur_group_name

    def open_group(self, key: str):
        self._cur_group = self._design_db[key]
        self._cur_group_name = key

    def keys(self):
        return self._cur_group.keys()

    def values(self):
        return self._cur_group.values()

    def items(self):
        return self._cur_group.items()

    @staticmethod
    def _get_hashable_key(item: Dict[str, Any]) -> ImmutableType:
        tbs = item['measurements'][0]['testbenches']
        tb = next(iter(tbs.values()))
        key = dict(
            schematic_params=item['schematic_params'],
            sim_params=tb['sim_params'],
        )
        return to_immutable(key)

    def __contains__(self, item: Dict[str, Any]):
        hashable_key = self._get_hashable_key(item)
        return hashable_key in self._cur_group

    def __getitem__(self, item: Dict[str, Any]):
        hashable_key = self._get_hashable_key(item)
        return self._cur_group[hashable_key]

    def __setitem__(self, key, value):
        hashable_key = self._get_hashable_key(key)
        self._cur_group[hashable_key] = value

    def __delitem__(self, key):
        del self._cur_group[key]

    def __repr__(self):
        return repr(self._design_db)


class TopLevelDesigner:

    def __init__(self, bprj: BagProject, spec_file: str = '',
                 spec_dict: Optional[Dict[str, Any]] = None,
                 sch_db: Optional[ModuleDB] = None, lay_db: Optional[TemplateDB] = None) -> None:

        if spec_dict:
            self._specs = spec_dict
        elif spec_file:
            self._specs = read_yaml(spec_file)

        dsn_params = self._specs['dsn_params']
        self.output_params = dsn_params['output']
        self.input_params = dsn_params['input']
        self._root_dir = Path(self._specs['root_dir']).resolve()

        self._prj = bprj

        pdir = Path(__file__).parent
        config_file = Path(str(pdir), 'output_driver.yaml')
        self._config = read_yaml(config_file)

        if sch_db is None:
            self._sch_db = ModuleDB(bprj.tech_info, self._specs['impl_lib'], prj=bprj)
        else:
            self._sch_db = sch_db

        if lay_db is None:
            self._lay_db = TemplateDB(bprj.grid, self._specs['impl_lib'], prj=bprj)
        else:
            self._lay_db = lay_db

        self._design_db = DesignDB(groups=[f'step{i+1}' for i in range(4)])

        # for later access, to config files store them in a dictionary
        self._digital_db_files = {}
        for k, v in self._specs['digital_db_files'].items():
            self._digital_db_files[k] = v[0]

    def design(self) -> Any:
        out_tr_widths, min_nsegs = self._get_em_specs()
        rout_ucell = 6 * self.output_params['rout']
        rout_weak = self.output_params['rout_weak']
        done = False
        while not done:
            # driver
            weak_pupd_params = self._design_ucell(rout_weak, min_nsegs, weak=True)
            ucell_pupd_params = self._design_ucell(rout_ucell, min_nsegs)
            ucell_lvl_shifter_params = self._design_lvl_shifter(ucell_pupd_params)

            pprint.pprint(weak_pupd_params)
            pprint.pprint(ucell_pupd_params)
            pprint.pprint(ucell_lvl_shifter_params)
            pdb.set_trace()
            done = True

    def gen_db(self) -> None:
        for config in self._specs['digital_db_files'].values():
            config_fname = config[0]
            for mos_type in config[1]:
                DigitalDB(self._prj, config_fname, mos_type=mos_type, load_from_file=False)

    def _get_em_specs(self, **kwargs: Any) -> Tuple[Dict[int, int], int]:
        """
        Computes the minimum number of fingers and wire widths required for the output driver
        inverter that complies with EM rules of the PDK.
        :param kwargs:
            Keyword arguments passed to grid.get_min_track_width. (i.e. temperature etc.)
        :return:
            tr_widths: a dictionary that should get passed to TrackManager's config view for
            layout stuff. It represents the wire widths of output wire type.
            min_nseg: minimum number of fingers that should be used for output driver
        """
        freq = self.input_params['freq']
        cout = self.input_params['cout']
        vdd = self.input_params['vdd_io']
        divider = 3  # in the worst case for current density only three unit cells are activated

        # For cds_ff_mpt these values don't change anything
        ipeak_targ = cout * vdd * freq / divider
        idc_targ = 0.0
        irms_targ = ipeak_targ / 2 ** 0.5

        grid = self._prj.grid

        # step 1: find the minimum width for metal 3 connected to output
        # if the minimum is more than what's allowed BAG's API will break it up to multiple
        # narrower wires when they get instantiated, so do not worry about this width being
        # larger than the maximum allowed by PDK
        m3_width = grid.get_min_track_width(layer_id=3, idc=idc_targ, iac_rms=irms_targ,
                                            iac_peak=ipeak_targ, **kwargs)
        # step 2: find the minimum width for metal 2 connected to output
        m2_width = grid.get_min_track_width(layer_id=2, idc=idc_targ, iac_rms=irms_targ,
                                            iac_peak=ipeak_targ, **kwargs)
        # step 3: find the minimum number of drain connections and from that find the minimum
        # number of fingers of transistors (ds pitch is fixed)
        m1_width = grid.get_min_track_width(layer_id=1, idc=idc_targ, iac_rms=irms_targ,
                                            iac_peak=ipeak_targ, **kwargs)

        tr_widths = {1: 1, 2: m2_width, 3: m3_width}
        min_segs = 2 * m1_width - 1

        return tr_widths, min_segs

    def _design_ucell(self, rout: float, min_segs: int = 1, weak: bool = False) -> Dict[str, Any]:
        """
        Designs a unit cell pseudo inverter where gate of p and nmos are not connected.
        If weak is True, the gates are directly connected to control signals, otherwise,
        the pmos gate is connected to a NAND and the nmos gate is connected to a NOR.
        The design is done such that the pull up and pull downs resistances independently satisfy
        rout and EM requirements. In case of weak = False, the NAND and NOR are sized such that
        rise time and fall time match according to the specification
        :param rout:
            The requirement for output resistance. It is assumed to be symmetric for pull up and
            pull down.
        :param min_segs:
            Minimum number of segments needed for satisfying EM rules.
        :param weak:
            If True there is no NAND/NOR at the input
        :return:
            new_params: dictionary of parameters that specify unit cell's physical dimensions
        """
        self._design_db.open_group('step1')
        cout = self.input_params['cout']
        vdd = self.input_params['vdd_io']
        nfin_ref = self.input_params['nfin_min']
        pfin_ref = self.input_params['pfin_min']
        nseg_ref = max(self.input_params['nseg_min'], min_segs)
        pseg_ref = max(self.input_params['pseg_min'], min_segs)

        rel_trf_matching: float = self.input_params['rel_trf_matching']
        fanout: float = self.input_params['nand_nor_fanout']

        lch_ref = self.input_params['lch']
        stack_ref = 1
        tper = max(10 * rout * cout, 20e-9)

        ref_params = dict(
            lch=lch_ref,
            seg_p=pseg_ref,
            seg_n=nseg_ref,
            w_p=pfin_ref,
            w_n=nfin_ref,
            stack_p=stack_ref,
            stack_n=stack_ref,
        )
        rout_pu_ref, rout_pd_ref, tper = self._get_rout_with_no_tper(cout, vdd, tper, ref_params,
                                                                     minim=weak)
        if rout > min(rout_pu_ref, rout_pd_ref):
            # we don't change lch from minimum because it might not be preferable for other
            # blocks, and we can't have multiple lch in a single MOSBase
            stack_init = int(rout // min(rout_pu_ref, rout_pd_ref)) + 1
            for stack in range(stack_init, stack_init + 5, 1):
                # another assumption is that stack for p and n should be equal, because of the
                # layout generator
                ref_params['stack_p'] = stack
                ref_params['stack_n'] = stack
                rout_pu_ref, rout_pd_ref, tper = self._get_rout_with_no_tper(cout, vdd, tper,
                                                                             ref_params,
                                                                             minim=weak)
                if rout < min(rout_pu_ref, rout_pd_ref):
                    # if still lower than rout continue
                    break
            if rout > min(rout_pu_ref, rout_pd_ref):
                # if still lower than rout continue
                raise ValueError('cannot design weak pu pd')
            # find the right number of fins and segments
            pupd_params = self._size_pupd_ucell(cout, vdd, tper, rout, rout, rout_pd_ref,
                                                rout_pu_ref, ref_params, round_down=weak)
        else:
            # number of stacks is fine, now find the right number of fins and segments
            pupd_params = self._size_pupd_ucell(cout, vdd, tper, rout, rout, rout_pd_ref,
                                                rout_pu_ref, ref_params)

        if weak:
            return pupd_params

        self._design_db.open_group('step2')
        nand_nor_results = self._design_nand_nor(tper, rel_trf_matching, fanout, pupd_params)

        nand_params = nand_nor_results['nand_params']
        nor_params = nand_nor_results['nor_params']

        ucell_dsn = dict(
            nand=nand_params,
            nor=nor_params,
            pupd=pupd_params
        )
        return ucell_dsn

    def _size_pupd_ucell(self, cout: float, vdd: float, tper: float, rout_pd: float,
                         rout_pu: float, rout_pd_ref: float,
                         rout_pu_ref: float, ref_params: Dict, round_down: bool = False):

        '''
        Simulation based design method for sizing pull up and pull down ucells (weak or strong)

        :param cout:
        :param vdd:
        :param tper:
        :param rout_pd:
        :param rout_pu:
        :param rout_pd_ref:
        :param rout_pu_ref:
        :param ref_params:
        :param round_down:
        :return:
        '''

        round_fn = floor if round_down else ceil
        params = ref_params.copy()

        w_n_max = self.input_params['nfin_max']
        nseg_ref = ref_params['seg_n']
        w_n_ref = ref_params['w_n']

        nseg = round_fn(rout_pd_ref / rout_pd * nseg_ref)
        # this could be done here but let's make aspect ratio closest to 1, considering w_n_max
        new_w_n = min(w_n_max, floor(sqrt(nseg * w_n_ref)))
        sim_params = dict(vdd=vdd, cout=cout, tper=tper)
        if new_w_n != params['w_n']:
            params['w_n'] = new_w_n
            res = self._simulate_and_process(sim_params, params, step=1, minim=round_down)
            rout_pd_ref = res['pulldown_res']
            nseg = round_fn(rout_pd_ref / rout_pd * nseg_ref)

        w_p_max = self.input_params['pfin_max']
        pseg_ref = ref_params['seg_p']
        w_p_ref = ref_params['w_p']

        pseg = round_fn(rout_pu_ref / rout_pu * pseg_ref)
        new_w_p = min(w_p_max, floor(sqrt(pseg * w_p_ref)))
        if new_w_p != params['w_p']:
            params['w_p'] = new_w_p
            res = self._simulate_and_process(sim_params, params, step=1, minim=round_down)
            rout_pu_ref = res['pullup_res']
            pseg = round_fn(rout_pu_ref / rout_pu * pseg_ref)

        # adjusting seg_n and seg_p so that the difference is even
        if abs(nseg - pseg) % 2 != 0:
            if round_down:
                if nseg > pseg:
                    nseg -= 1
                else:
                    pseg -= 1
            else:
                if nseg < pseg:
                    nseg += 1
                else:
                    pseg += 1

        params['seg_n'] = nseg
        params['seg_p'] = pseg

        res = self._simulate_and_process(sim_params, params, step=1, minim=round_down)
        rout_pu_ver = res['pullup_res']
        rout_pd_ver = res['pulldown_res']

        print('[Unit Cell Design Done]')
        print(f'rpu={rout_pu_ver}, rpd={rout_pd_ver}')
        print('Params:')
        pprint.pprint(params)

        return params

    def _design_nand_nor(self, tper, tr_tf_err, fanout, pupd_params):

        seg_n = pupd_params['seg_n']
        seg_p = pupd_params['seg_p']
        w_p = pupd_params['w_p']
        w_n = pupd_params['w_n']

        cout = self.input_params['cout']
        vdd = self.input_params['vdd_io']

        answer = None
        for stack_ratio in np.arange(1, 2, 0.1):
            for p_n_ratio in np.arange(1, 2, 0.1):
                stackn_ratio = stack_ratio
                stackp_ratio = stack_ratio
                nor_x1_wn = 1
                nor_x1_wp = ceil(p_n_ratio * stackp_ratio)
                nand_x1_wn = ceil(stackn_ratio)
                nand_x1_wp = ceil(p_n_ratio)

                nor_le = (nor_x1_wn + nor_x1_wp) / nor_x1_wn / (p_n_ratio + 1)
                nand_le = (nand_x1_wn + nand_x1_wp) / nor_x1_wn / (p_n_ratio + 1)

                # cout / cin * LE = effective_fanout
                nor_size_init = ceil(seg_n * w_n / (nor_x1_wp + nor_x1_wn) * nor_le / fanout)
                nand_size_init = ceil(seg_p * w_p / (nand_x1_wn + nand_x1_wp) * nand_le / fanout)

                ref_params = deepcopy(self._config['step_2']['schematic_params'])
                nand_ref_params = dict(
                    seg=nand_size_init,
                    lch=pupd_params['lch'],
                    w_p=nand_x1_wp,
                    w_n=nand_x1_wn,
                )
                nor_ref_params = dict(
                    seg=nor_size_init,
                    lch=pupd_params['lch'],
                    w_p=nor_x1_wp,
                    w_n=nor_x1_wn,
                )

                ref_params['nand_params'].update(**nand_ref_params)
                ref_params['nor_params'].update(**nor_ref_params)
                ref_params['output_stage_params'].update(**pupd_params)

                sim_params = dict(vdd=vdd, cout=cout, tper=tper)
                results = self._simulate_and_process(sim_params, ref_params, step=2)
                tf = results['tf']
                tr = results['tr']
                done = ((max(tf, tr) - min(tf, tr)) / min(tf, tr)) < tr_tf_err
                change_nand = tf < tr

                side_key = 'nand_params' if change_nand else 'nor_params'
                low = ref_params[side_key]['seg']
                hi = low * 5
                iterator = BinaryIterator(low, hi, step=1)
                info = None
                if not done:
                    print("[info] rise/fall time matching got violated, adjusting the ratios ...")
                    while iterator.has_next():
                        new_seg = iterator.get_next()
                        if change_nand:
                            # tr is more so we should lower it by upsizing the nand
                            nand_ref_params['seg'] = new_seg
                            ref_params['nand_params'].update(**nand_ref_params)
                        else:
                            # tf is more so we should lower it by upsizing the nor
                            nor_ref_params['seg'] = new_seg
                            ref_params['nor_params'].update(**nor_ref_params)

                        results = self._simulate_and_process(sim_params, ref_params, step=2)
                        tf = results['tf']
                        tr = results['tr']

                        done = ((max(tf, tr) - min(tf, tr)) / min(tf, tr)) < tr_tf_err
                        if done:
                            iterator.save_info((new_seg, tf, tr))
                            iterator.down()
                        elif not change_nand ^ (tr <= tf):
                            iterator.down()
                        else:
                            iterator.up()

                    info = iterator.get_last_save_info()
                    if info:
                        _, tf, tr = info

                if done or info:
                    answer = dict(
                        nand_params=ref_params['nand_params'],
                        nor_params=ref_params['nor_params'],
                        tf=tf,
                        tr=tr,
                    )
                    pprint.pprint(answer)
                    return answer

        if not answer:
            raise ValueError('Adjustment algorithm sucks ...')

    def _design_lvl_shifter(self, ucell_pupd_params):
        fout = 4
        nand_params = ucell_pupd_params['nand']
        nor_params = ucell_pupd_params['nor']

        seg_nand = nand_params['seg']
        wn_nand = nand_params['w_n']
        wp_nand = nand_params['w_p']
        seg_nor = nor_params['seg']
        wn_nor = nor_params['w_n']
        wp_nor = nor_params['w_p']

        nand_pmos_db_config = self._digital_db_files['stack1_config_file']
        nand_nmos_db_config = self._digital_db_files['stdgates_pupd_file']
        nor_nmos_db_config = self._digital_db_files['stack1_config_file']
        nor_pmos_db_config = self._digital_db_files['stdgates_pupd_file']

        nand_pmos_db = DigitalDB(self._prj, nand_pmos_db_config, mos_type='pmos',
                                 load_from_file=True)
        nand_nmos_db = DigitalDB(self._prj, nand_nmos_db_config, mos_type='nmos',
                                 load_from_file=True)
        nor_nmos_db = DigitalDB(self._prj, nor_nmos_db_config, mos_type='nmos',
                                load_from_file=True)
        nor_pmos_db = DigitalDB(self._prj, nor_pmos_db_config, mos_type='pmos',
                                load_from_file=True)

        env = 'tt_25'
        nand_pmos_params = nand_pmos_db.query(params_dict={'lch': 36, 'w_p': wp_nand},
                                              meas_type=TimingMeasType.DELAY_L2H,
                                              vdd=1,
                                              env=env)

        nand_nmos_params = nand_nmos_db.query(params_dict={'lch': 36, 'w_p': wn_nand},
                                              meas_type=TimingMeasType.DELAY_L2H,
                                              env=env)

        nor_nmos_params = nor_nmos_db.query(params_dict={'lch': 36, 'w_p': wn_nor},
                                            meas_type=TimingMeasType.DELAY_L2H,
                                            vdd=1,
                                            env=env)

        nor_pmos_params = nor_pmos_db.query(params_dict={'lch': 36, 'w_p': wp_nor},
                                            meas_type=TimingMeasType.DELAY_L2H,
                                            env=env)
        nand_cin = seg_nand * (nand_nmos_params['cg'] + nand_pmos_params['cg'])
        nor_cin = seg_nor * (nor_nmos_params['cg'] + nor_pmos_params['cg'])

        cload_lvl_shifter = 6 * (nand_cin + nor_cin)
        next_ucell_inverter = self._design_ucell_inverter(vdd=1)

        ninv_segs = ceil(cload_lvl_shifter / fout / next_ucell_inverter['cin'])
        cin_inv = ninv_segs * next_ucell_inverter['cin']
        ucell_lvl_shifter = self._design_ucell_lvl_shifter(rst_ratio=2, fout=fout,
                                                           cload_targ=cin_inv, kratio=6)

        # arranging things in a dictionary
        sch_params = self._config['step_3']['schematic_params']
        next_inv_params = [dict(seg=1, lch=36, w_p=next_ucell_inverter['wp'],
                                w_n=next_ucell_inverter['wn'], th_p='standard',
                                th_n='standard',
                                stack=1)]
        lvl_shifter_params = deepcopy(sch_params['lvl_shifter_params'])
        lvl_shifter_params['seg_dict'] = dict(pd=ucell_lvl_shifter['w1'],
                                              pu=ucell_lvl_shifter['w2'],
                                              rst=ucell_lvl_shifter['w1'] * 2,)
        lvl_shifter_params['inv_params']['inv_params'] = next_inv_params

        # logical effort design method
        cin_lvl_shifter = ucell_lvl_shifter['cin']
        prev_ucell_inverter = self._design_ucell_inverter(vdd=0.8)

        tot_fout = cin_lvl_shifter / prev_ucell_inverter['cin']
        n_prev_inv = log(tot_fout) / log(fout)
        n_prev_inv = ceil(n_prev_inv / 2.) * 2
        eff_fout = tot_fout ** (1 / n_prev_inv)

        inv_params_temp = dict(seg=1, lch=36, w_p=prev_ucell_inverter['wp'],
                               w_n=prev_ucell_inverter['wn'], th_p='standard',
                               th_n='standard',
                               stack=1)

        prev_inv_params = [inv_params_temp]
        seg = 1
        for i in range(1, n_prev_inv):
            seg = round(eff_fout * seg)
            inv_params_temp['seg'] = seg
            prev_inv_params.append(inv_params_temp)

        return dict(
            lvl_shifter_params=lvl_shifter_params,
            inv_params=dict(inv_params=inv_params_temp),
        )

    def _design_ucell_inverter(self, vdd):
        db_config = self._digital_db_files['stack1_config_file']

        nmos_db = DigitalDB(self._prj, db_config, mos_type='nmos',
                            load_from_file=True)
        pmos_db = DigitalDB(self._prj, db_config, mos_type='pmos',
                            load_from_file=True)

        env = 'tt_25'
        wn = 1
        nparams = nmos_db.query(params_dict={'lch': 36, 'w_p': wn},
                                meas_type=TimingMeasType.DELAY_L2H,
                                vdd=vdd,
                                env=env)

        pparams = pmos_db.query(params_dict={'lch': 36, 'w_p': wn},
                                meas_type=TimingMeasType.DELAY_H2L,
                                vdd=vdd,
                                env=env)

        pn_ratio = pparams['res'] / nparams['res']
        wp = wn * pn_ratio

        def cost(x_: np.ndarray):
            xhat = np.round(x_)
            return np.sqrt(np.sum((x_ - xhat) ** 2))

        tot_cost = np.inf
        params = {'wn': wn, 'wp': wp}
        for wn_inv in range(1, 8, 1):
            wp_inv = wp * wn_inv
            new_cost = cost(np.array([wn_inv, wp_inv]))
            if new_cost < tot_cost:
                params['wn'] = round(wn_inv)
                params['wp'] = round(wp_inv)
                tot_cost = new_cost

        fn_par = nmos_db.query(params_dict={'lch': 36, 'w_p': params['wn']},
                               meas_type=TimingMeasType.DELAY_L2H,
                               vdd=vdd,
                               env=env)

        fp_par = nmos_db.query(params_dict={'lch': 36, 'w_p': params['wp']},
                               meas_type=TimingMeasType.DELAY_H2L,
                               vdd=vdd,
                               env=env)
        params.update(**dict(
            cin=fn_par['cg'] + fp_par['cg'],
            cout=fn_par['cd'] + fp_par['cd'],
            rout=0.5 * (fn_par['res'] + fp_par['res']),
        ))
        return params

    def _design_ucell_lvl_shifter(self, rst_ratio, fout, cload_targ, kratio):
        stack2_config_file = self._digital_db_files['stack2_config_file']
        stack1_config_file = self._digital_db_files['stack1_config_file']

        input_db = DigitalDB(prj, stack2_config_file, mos_type='nmos', load_from_file=True)
        p_db = DigitalDB(prj, stack1_config_file, mos_type='pmos', load_from_file=True)
        n_db = DigitalDB(prj, stack1_config_file, mos_type='nmos', load_from_file=True)

        params_dict = dict(
            lch=36,
            w_p=2,
        )
        env = 'tt_25'
        input_params = input_db.query(params_dict, TimingMeasType.DELAY_L2H, env=env)
        inv_pmos_params = p_db.query(params_dict, TimingMeasType.DELAY_H2L, vdd=0.8, env=env)
        inv_nmos_params = n_db.query(params_dict, TimingMeasType.DELAY_H2L, vdd=0.8, env=env)
        rst_nmos_params = n_db.query(params_dict, TimingMeasType.DELAY_H2L, vdd=0.8, env=env)
        xcoupled_params = p_db.query(params_dict, TimingMeasType.DELAY_H2L, vdd=1, env=env)

        rn = input_params['res']
        rp = xcoupled_params['res']
        cd_in = input_params['cd']
        cg_in = input_params['cg']
        cd_rst = rst_nmos_params['cd']
        cg_xp = xcoupled_params['cg']
        cd_xp = xcoupled_params['cd']

        cg_inv = inv_nmos_params['cg'] + inv_pmos_params['cg']
        cd_inv = inv_nmos_params['cd'] + inv_pmos_params['cd']
        rp_inv = inv_pmos_params['res']

        alpha = 1   # fudge factor for the fight between the right nmos and xcoupled pmos
        k1 = 1      # fudge factor for ttop1
        k2 = 1      # fudge factor for tbot1
        k3 = 1      # fudge factor for ttop2
        k4 = 1      # fudge factor for tbot2 (not really useful! alpha takes care of it)
        # # fudge factor coefficient for the fight between the right nmos and xcoupled pmos
        done = False
        fparams = None
        while not done:
            params = {}
            def t1_l2h(w1_loc, w2_loc):
                td = k3 / (w1_loc / rn - w2_loc / rp) * \
                     (w1_loc * cd_in + rst_ratio * w1_loc * cd_rst + w2_loc * (cg_xp + cd_xp))
                return td

            def t2_l2h(w1_loc, w2_loc, cl):
                td = k4 * rp / w2_loc * \
                     (w2_loc * (cg_xp + cd_xp) + w1_loc * cd_in + rst_ratio * w1_loc * cd_rst + cl)
                return td

            def tdl2h(w1_loc, w2_loc, cl):
                td1 = t1_l2h(w1_loc, w2_loc)
                td2 = t2_l2h(w1_loc, w2_loc, cl)
                print('l2h: td1 = ', td1)
                print('l2h: td2 = ', td2)
                _tdl2h = td1 + td2
                return _tdl2h

            def t1_h2l(w1_loc, winv_loc):
                td = k1 * rp_inv / winv_loc * (winv_loc * cd_inv + w1_loc * cg_in)
                return td

            def t2_h2l(w1_loc, w2_loc, cl, alpha_loc):
                td = k2 / (w1_loc / rn - 1 / alpha_loc * w2_loc / rp) * \
                     (w2_loc * (cg_xp + cd_xp) + w1_loc * cd_in + rst_ratio * w1_loc * cd_rst + cl)
                return td

            def tdh2l(w1_loc, w2_loc, winv_loc, cl, alpha_loc):
                td1 = t1_h2l(w1_loc, winv_loc)
                td2 = t2_h2l(w1_loc, w2_loc, cl, alpha_loc)
                _tdh2l = td1 + td2
                print('h2l: td1 = ', td1)
                print('h2l: td2 = ', td2)
                return _tdh2l

            unit_param_vec = None
            cload = cload_targ
            # plt.clf()
            for winv in range(1, 10, 1):
                # k = rp / rn * alpha / (alpha + 1)
                k = 1 / kratio

                def fzero(w1p) -> float:
                    return t1_l2h(w1p, k * w1p) - t1_h2l(w1p, winv) + \
                           (t2_l2h(w1p, k * w1p, cload_targ) - t2_h2l(w1p, k * w1p,
                                                                      cload_targ, alpha))

                # x = np.linspace(1, 100, 1000)
                # plt.plot(x, fzero(x))
                w1 = cast(float, brentq(fzero, 1, 100))
                w2 = k * w1
                cin = (w1 * cg_in + winv * cg_inv)

                rpu = rp / w2
                rpd = 1 / (w1 / rn - 1 / alpha * w2 / rp)

                print('before scaling ...')
                print('-' * 30)
                print(f'winv = {winv}')
                print(f'w1 = {w1}')
                print(f'w2 = {w2}')
                print(f'cload = {cload}')
                print(f'tdh2l = {tdh2l(w1, w2, winv, cload, alpha)}')
                print(f'tdl2h = {tdl2h(w1, w2, cload)}')
                print(f'rpu = {rpu}, rpd = {rpd}')
                print(f'cin = {cin}')
                print(f'fout = {cload / cin}')
                unit_param_vec = np.array([winv, w1, w2])
                if (cload / cin) < fout:
                    params = dict(w1=w1, w2=w2, winv=winv,
                                  cload=cload_targ, fout=cload / cin, cin=cin)
                    break
            print('after rounding ...')
            # plt.show()

            def cost(x_: np.ndarray):
                xhat = np.round(x_)
                return np.sqrt(np.sum((x_ - xhat) ** 2))
            tot_cost = np.inf
            new_params_vec = None
            opt_coef = 1
            for coef in np.arange(1, 2, 0.1):
                x = unit_param_vec * coef
                new_cost = cost(x)
                if new_cost < tot_cost:
                    new_params_vec = x
                    tot_cost = new_cost
                    opt_coef = coef
            winv, w1, w2 = np.round(new_params_vec).astype(int)
            cin = (w1 * cg_in + winv * cg_inv)
            cload *= opt_coef
            rpu = rp / w2
            rpd = 1 / (w1 / rn - 1 / alpha * w2 / rp)
            ttop1_calc = t1_l2h(w1, w2)
            ttop2_calc = t2_l2h(w1, w2, cload)
            ttop_calc = ttop1_calc + ttop2_calc
            tbot1_calc = t1_h2l(w1, winv)
            tbot2_calc = t2_h2l(w1, w2, cload, alpha)
            tbot_calc = tbot1_calc + tbot2_calc
            print(f'winv = {winv}')
            print(f'w1 = {w1}')
            print(f'w2 = {w2}')
            print(f'cload = {cload}')
            print(f'tdh2l = {tdh2l(w1, w2, winv, cload, alpha)}')
            print(f'tdl2h = {tdl2h(w1, w2, cload)}')
            print(f'rpu = {rpu}, rpd = {rpd}')

            ref_params = self._config['step_3']['schematic_params'].copy()
            seg_dict = ref_params['lvl_shifter_params']['seg_dict']
            seg_dict.update(**dict(pd=w1, pu=w2, rst=2*w1))
            inv_params = ref_params['inv_params']['inv_params'][0]
            inv_params['seg'] = winv
            sim_params = dict(cout=cload)
            results = self._simulate_and_process(sim_params, ref_params, step=3)
            k1 *= results['tbot1'] / tbot1_calc
            k2 *= results['tbot2'] / tbot2_calc
            k3 *= results['ttop1'] / ttop1_calc
            k4 *= results['ttop2'] / ttop2_calc

            pprint.pprint(results)
            print(f'k1 = {k1}')
            print(f'k2 = {k2}')
            print(f'k3 = {k3}')
            print(f'k4 = {k4}')

            def err(x, xhat, eps=1e-15):
                return abs(x-xhat) / (min(x, xhat) + eps)

            err_tol = 0.05
            if err(results['tbot'], results['ttop']) < err_tol:
                done = True
                winv = round(params['winv'])
                w1 = round(params['w1'])
                w2 = round(params['w2'])
                cin = (w1 * cg_in + winv * cg_inv)
                fparams = dict(w1=w1, w2=w2, winv=winv,
                               cload=cload_targ, fout=cload_targ / cin, cin=cin,
                               tdh2l=tdh2l(w1, w2, winv, cload_targ, alpha),
                               tdl2h=tdl2h(w1, w2, cload_targ))
        return fparams

    def _get_rout_with_no_tper(self, cout, vdd, tper_init,
                               ref_params, minim=False) -> Tuple[float, float, float]:
        # runs rout measurement and returns the tper that worked
        tper = tper_init
        while True:
            sim_params = dict(vdd=vdd, cout=cout, tper=tper)
            try:
                result = self._simulate_and_process(sim_params, ref_params, step=1, minim=minim)
                return result['pullup_res'], result['pulldown_res'], tper
            except TimeTooSmallError:
                tper *= 2
                print(f'[Failed] Simulation due to tper being small, trying tper = {tper} ...')

    def _simulate(self, step_params):
        sim = DesignManager(self._prj, spec_dict=step_params, sch_db=self._sch_db,
                            lay_db=self._lay_db)
        sim.characterize_designs(generate=True, measure=True, load_from_file=False, verbose=False)
        dsn_name = next(iter(sim.info.dsn_name_iter()))
        results = sim.get_result(dsn_name)
        return results

    def _get_dsn_man_params(self, sim_params, ref_params, mode, step):
        step_id = f'step_{step}'
        keys = self._config[step_id]['schematic_params'].keys()
        for key in keys:
            if key not in ref_params and not key.startswith('dumm'):
                raise KeyError(f'Missing key "{key}" in ref_params in _get_rout')

        step_params = deepcopy(self._config[step_id])
        meas_params = step_params['measurements'][0]
        tb_params = next(iter(meas_params['testbenches'].values()))
        tb_sim_params = tb_params['sim_params']
        step_params['impl_lib'] = self._specs['impl_lib']
        step_params['view_name'] = mode
        step_params['root_dir'] = self._specs['root_dir']
        step_params['env_list'] = self._specs['env_list']
        tb_sim_params.update(**sim_params)
        step_params['schematic_params'].update(**ref_params)
        return step_params

    def _simulate_and_process(self, sim_params: Dict[str, Any], ref_params: Dict[str, Any],
                              mode: str = 'schematic', step: int = 1,
                              minim=False) -> Dict[str, float]:

        min_max_fn = min if minim else max
        step_params = self._get_dsn_man_params(sim_params, ref_params, mode, step)
        self._design_db.open_group(f'step{step}')
        if step_params not in self._design_db:
            results = self._simulate(step_params)
            results = next(iter(results.values()))
        else:
            return self._design_db[step_params]

        # pprint.pprint(step_params)
        if step == 1:
            if hasattr(results['pullup_res'], '__iter__'):
                pu = min_max_fn(results['pullup_res'])
                pd = min_max_fn(results['pulldown_res'])
            else:
                pu = results['pullup_res']
                pd = results['pulldown_res']
            results = dict(pullup_res=pu, pulldown_res=pd)
        elif step == 2:
            pass
        elif step == 3:
            try:
                next(iter(results['ttop']))
                index = np.argmax(results['ttop'])
                for k, v in results.items():
                    results[k] = v[index]
            except TypeError:
                for k, v in results.items():
                    results[k] = float(v)
        else:
            raise ValueError('Unkown step number ....')

        if step_params not in self._design_db:
            self._design_db[step_params] = results
        return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--gen_db', action='store_true')
    pargs = parser.parse_args()

    local_dict = locals()
    if 'prj' not in local_dict:
        print('creating bag project')
        prj = BagProject()
    else:
        print('loading bag project')
        prj = local_dict['prj']

    pardir = Path(__file__).parent
    yaml_f = Path(str(pardir), 'output_driver_dsn_config.yaml')
    yaml_content: Dict[str, Any] = read_yaml(yaml_f)
    designer = TopLevelDesigner(prj, yaml_f)
    if pargs.gen_db:
        designer.gen_db()
    else:
        designer.design()
        pdb.set_trace()



