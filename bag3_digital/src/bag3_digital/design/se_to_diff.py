"""This package contains design class for SingleToDiff"""

from typing import Dict, Any, Tuple, Optional, Mapping, Union, Type, Sequence, cast

from pathlib import Path
from math import ceil, floor
import numpy as np
from copy import deepcopy

from bag.math import float_to_si_string
from bag.io.file import read_yaml

from xbase.layout.mos.base import MOSBasePlaceInfo
from xbase.layout.enum import MOSType

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB
from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_testbenches.design.base import DesignerBase
from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag3_digital.layout.stdcells.se_to_diff import SingleToDiff
from bag3_digital.layout.stdcells.gates import InvCore, InvChainCore, PassGateCore
from bag3_digital.design.digital_db.db import DigitalDB


class SingleToDiffDesigner(DesignerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)

        self._tb_params = dict(
            load_list=[('outp', 'cload'), ('outn', 'cload')],
            vbias_list=[('VDD', 'vdd')],
            dut_conns={
                'in': 'in',
                'outp': 'outp',
                'outn': 'outn',
                'VDD': 'VDD',
                'VSS': 'VSS'
            },
        )

    async def async_design(self, cout: float, cin: float, freq: float, vdd: float,
                           trf_in: float, sim_envs: Sequence[str], pinfo, tile_name: str = '',
                           unit_inv_cin: float = 0, cap_in_mm_specs: str = '',
                           **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Design a single-ended to differential splitter.
        """
        if unit_inv_cin == 0 and cap_in_mm_specs == '':
            raise ValueError("Either unit_inv_cin or cap_in_mm_specs must be provided")

        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, pinfo, tile_name)
        if not isinstance(pinfo, MOSBasePlaceInfo):
            # TODO: support tiling
            _, tinfo_table = pinfo
            pinfo: MOSBasePlaceInfo = tinfo_table[tile_name]

        if unit_inv_cin == 0:
            unit_inv_cin = await self._get_unit_inv_cin(pinfo, cap_in_mm_specs)
            print('unit_inv_cin', unit_inv_cin)

        # characterize
        tinv, gamma, tpg = await self.get_tinv_gamma_tpg(pinfo)

        # intrinsic delay/input cap/output res ratios of passgate to inverter
        t_ratio = tpg / tinv
        c_ratio = gamma  # cin_pg / cin_inv = cd / cg = gamma
        r_ratio = t_ratio / c_ratio

        # outp path: inv a1, pg a2, inv a3
        # outn path: inv b1, pg b2, inv b3
        c_ratio_F = cout / unit_inv_cin

        # total number of input unit inverters = a1 + b1
        seg_in = int(np.rint(cin / unit_inv_cin))
        print("seg_in", seg_in)

        b1_list = np.arange(start=1, stop=seg_in, step=1)
        a2 = None
        b1 = None
        td_min = np.inf
        for _b1 in reversed(b1_list):
            print('---')
            root0, root1 = np.roots([2 * c_ratio / (seg_in - _b1),
                                    - gamma - 2 * np.cbrt(c_ratio_F / _b1) + t_ratio +
                                    np.cbrt(_b1 * c_ratio_F ** 2) / (seg_in - _b1),
                                    np.cbrt(_b1 * c_ratio_F ** 2) * r_ratio])
            if np.iscomplex(root0) or np.iscomplex(root1):
                print('complex')
                continue

            if root0 < 0 and root1 < 0:
                print('negative')
                continue

            if root0 > 0 and root1 > 0:
                _a2 = int(np.rint(min(root0, root1)))
            else:
                _a2 = int(np.rint(max(root0, root1)))
            td = 2 * (gamma + np.cbrt(c_ratio_F / _b1))
            if td < td_min:
                # TODO: Fix code so that if hit cases like this rerun the design with the assumption that the device
                #  that is too small is fixed to 1 segment, then size everything else accordingly to match delay.
                b1 = _b1 if _b1 > 0 else 1
                a2 = _a2 if _a2 > 0 else 1
                print(f'b1={b1}')
                print(f'a2={a2}')
                print(f'td={td}')
                td_min = td

        if a2 is None:
            raise ValueError('Failed to find solution')

        tbm_specs = dict(
            sim_envs=sim_envs,
            thres_lo=0.1,
            thres_hi=0.9,
            in_pwr='vdd',
            tstep=None,
            sim_params=dict(
                vdd=vdd,
                cload=cout,
                tbit=1.0 / freq,
                trf=trf_in,
            ),
            save_outputs=['in', 'outp', 'outn', 'XDUT.midn_pass1', 'XDUT.midp'],
            rtol=1e-8,
            atol=1e-22,

            out_invert=False,
            print_delay_list=[
                ('in', 'outp'),
                ('in', 'outn', True),
                ('in', 'XDUT.midn_pass1', True),
                ('in', 'XDUT.midp')
            ],
            print_trf_list=['outp', 'outn'],

            sim_options={},
        )
        tbm = cast(CombLogicTimingTB, self.make_tbm(CombLogicTimingTB, tbm_specs))

        best_worst_dev = 101
        best_designed_params = None
        best_designed_dict = None

        # for b1_t in [b1 - 1, b1, b1 + 1]:
        for b1_t in [b1]:
            # for a2_t in [a2 - 1, a2, a2 + 1]:
            for a2_t in [a2]:
                print('------------------------------------------------------------------')
                fanout_per_stage = np.cbrt(c_ratio_F / b1_t)
                b2 = int(np.rint(b1_t * fanout_per_stage))
                a3 = b3 = int(np.rint(b2 * fanout_per_stage))

                a1 = seg_in - b1_t

                print(a1, a2_t, a3)
                print(b1_t, b2, b3)
                print('Estimated delay of path a:', gamma + 2 * c_ratio * a2_t / a1 + t_ratio
                      + r_ratio * a3 / a2_t + a3 / a1)
                print('Estimated delay of path b:', 2 * (gamma + fanout_per_stage))

                subblock_params = self.update_params(a1, a2_t, a3, b1_t, b2, b3)

                dut_params = dict(
                    pinfo=pinfo,
                    **subblock_params,
                )
                for opt_param in ['is_guarded', 'swap_tiles', 'vertical_out', 'sig_locs']:
                    if opt_param in kwargs:
                        dut_params[opt_param] = kwargs[opt_param]

                design_dict, worst_dev = await self.verify_design(f'verify_b1_t_{b1_t}_a2_t_{a2_t}',
                                                                  dut_params, tbm)

                if worst_dev < best_worst_dev:
                    best_worst_dev = worst_dev
                    best_designed_params = subblock_params
                    best_designed_dict = design_dict

        return best_designed_params, best_designed_dict

    async def verify_design(self, sim_id: str, dut_params: Mapping[str, Any],
                            tbm: CombLogicTimingTB) -> Tuple[Dict[str, Any], float]:
        gen_params = dict(
            cls_name=SingleToDiff.get_qualified_name(),
            draw_taps=True,
            params=dut_params,
        )
        dut = await self.async_new_dut('unit_cell', STDCellWrapper, gen_params)

        sim_results = await self.async_simulate_tbm_obj(sim_id, dut, tbm, self._tb_params)
        sim_data = sim_results.data

        ptdr, ptdf = CombLogicTimingTB.get_output_delay(sim_data, tbm.specs, 'in', 'outp',
                                                        out_invert=False)
        ntdr, ntdf = CombLogicTimingTB.get_output_delay(sim_data, tbm.specs, 'in', 'outn',
                                                        out_invert=True)
        in_rise_dev = (ptdr - ntdf) / ntdf * 100
        worst_rise_dev = max(abs(in_rise_dev))
        in_fall_dev = (ptdf - ntdr) / ntdr * 100
        worst_fall_dev = max(abs(in_fall_dev))
        worst_dev = max(worst_rise_dev, worst_fall_dev)

        out_trfs = CombLogicTimingTB.get_output_trf(sim_data, tbm.specs, 'outp') + \
                   CombLogicTimingTB.get_output_trf(sim_data, tbm.specs, 'outn')
        slowest_trf = max(map(lambda arr: np.mean(arr), out_trfs))
        trf_to_td = np.log(2) / \
                    np.log((1 - tbm.specs['thres_lo']) / (1 - tbm.specs['thres_hi']))
        last_stage_delay = trf_to_td * slowest_trf

        design_dict = dict(
            in_rise_outp=ptdr,
            in_rise_outn=ntdf,
            in_rise_dev=in_rise_dev,
            in_fall_outp=ptdf,
            in_fall_outn=ntdr,
            in_fall_dev=in_fall_dev,
            last_stage_delay=last_stage_delay,
            sim_envs=sim_data.sim_envs,
        )

        print(f'{"sim_env"}{"in_rise_outp":>20}{"in_rise_outn":>20}{"in_rise_dev":>20}'
              f'{"in_fall_outp":>20}{"in_fall_outn":>20}{"in_fall_outp":>20}')
        for idx, env in enumerate(sim_data.sim_envs):
            print(f'{env}{ptdr[idx]:>20.3g}{ntdf[idx]:>20.3g}{in_rise_dev[idx]:>20.3g}'
                  f'{ptdf[idx]:>20.3g}{ntdr[idx]:>20.3g}{in_fall_dev[idx]:>20.3g}')
        print(f'Worst deviation: {worst_dev}')

        return design_dict, worst_dev

    async def get_tinv_gamma_tpg(self, pinfo):
        # setup parameters
        wn, wp = _get_default_width(pinfo)
        # Assumes same threshold for all rows
        intent = pinfo.get_row_place_info(0).row_info.threshold
        vdd = 0.8
        tper = 1/1e9

        # setup TBM
        tb_params = dict(
            load_list=[('out', 'cload')],
            dut_conns={'in': 'in',
                       'out': 'out',
                       'VDD': 'VDD',
                       'VSS': 'VSS',
                       'mid': "mid",
                       'en': 'VDD',
                       'enb': 'VSS',
                       's': 'in',
                       'd': 'out'},
        )

        sim_params = dict(
            tbit=tper,
            trf=1.0e-12,
            vdd=vdd,
            cload=0,
        )

        tbm_specs=dict(
            sch_params=tb_params,
            sim_envs=['tt_25'],
            tstep=None,
            thres_lo=0.1,
            thres_hi=0.9,
            stimuli_pwr='vdd',
            rtol=1e-8,
            atol=1e-22,
            save_outputs=['in', 'out', 'mid'],
            sim_params=sim_params,
        )
        tbm = self.make_tbm(CombLogicTimingTB, tbm_specs)

        # setup self-loaded inverter DUT
        dut_params = dict(
            pinfo=pinfo,
            seg = 1,
            w_p=wp,
            w_n=wn
        )
        dut = await self.async_new_dut("inv_test", InvCore, dut_params, extract=False)

        # get result
        result = await self.async_simulate_tbm_obj(f'tinv_1', dut, tbm, tb_params)
        t1_r, t1_f = CombLogicTimingTB.get_output_delay(result.data, tbm_specs, 'in', 'out', True)
        print(t1_r, t1_f)
        print((t1_r - t1_f) / (t1_r + t1_f) * 2)
        tp1 = (t1_r + t1_f) / 2

        # setup FO1 dut
        dut_params = dict(
            pinfo=pinfo,
            seg_list = [1] * 2,
            w_p=wp,
            w_n=wn,
            export_pins=True
        )
        dut = await self.async_new_dut("inv_chain_test", InvChainCore, dut_params, extract=False)

        # get result
        result = await self.async_simulate_tbm_obj(f'tinv_2', dut, tbm, tb_params)
        t2_r, t2_f = CombLogicTimingTB.get_output_delay(result.data, tbm_specs, 'in', 'mid', True)
        print(t2_r, t2_f)
        print((t2_r - t2_f) / (t2_r + t2_f) * 2)

        tp2 = (t2_r + t2_f) / 2
        tinv = tp2 - tp1
        gamma = tp1 / tinv
        print("tinv", tinv[0])
        print("gamma", gamma[0])

        # setup PG dut
        dut_params = dict(
            pinfo=pinfo,
            seg=1,
            w_p=wp,
            w_n=wn,
        )
        dut = await self.async_new_dut("pg_test", PassGateCore, dut_params, extract=False)

        result = await self.async_simulate_tbm_obj(f'tpg', dut, tbm, tb_params)
        t3_r, t3_f = CombLogicTimingTB.get_output_delay(result.data, tbm_specs, 'in', 'out', False)
        tp3 = (t3_r + t3_f) / 2

        tpg = tp3[0]  # TODO: improve definition
        print('tpg', tpg)

        return tinv[0], gamma[0], tpg

    async def _get_unit_inv_cin(self, pinfo: Mapping[str, Any], cap_in_mm_specs: str) -> float:

        # setup parameters
        wn, wp = _get_default_width(pinfo)
        # Assumes same threshold for all rows
        intent = pinfo.get_row_place_info(0).row_info.threshold

        # setup MM
        mm_top_specs = read_yaml(cap_in_mm_specs)
        mm_specs = mm_top_specs['meas_params']
        mm = self.make_mm(CapDelayMatch, mm_specs)

        # Setup DUT
        dut_params = dict(
            pinfo=pinfo,
            seg=1,
            w_p=wp,
            w_n=wn
        )
        dut = await self.async_new_dut("inv_test", InvCore, dut_params, extract=False)

        # get result
        result = await self.async_simulate_mm_obj(f'cap_in', dut, mm)
        cap = (result.data['cap_rise'] + result.data['cap_fall']) / 2
        
        return cap

    @staticmethod
    def update_params(a1: int, a2: int, a3: int, b1: int, b2: int, b3: int) -> Dict[str, Any]:
        return dict(
            invp_params_list=[{'seg': a1}, {'seg': a3}],
            invn_params_list=[{'seg': b1}, {'seg': b2}, {'seg': b3}],
            pg_params={'seg': a2}
        )


def _get_default_width(pinfo: MOSBasePlaceInfo) -> Tuple[int, int]:
    wn, wp = [], []
    for row_place_info in map(pinfo.get_row_place_info, range(pinfo.num_rows)):
        w = row_place_info.row_info.width
        if row_place_info.row_info.row_type is MOSType.nch:
            wn.append(w)
        elif row_place_info.row_info.row_type is MOSType.pch:
            wp.append(w)
    # In the case that there are multiple NMOS or PMOS rows, this function returns the
    # most strict constraint. Typically, the width ends up being the same anyway.
    if len(wn) > 1:
        wn = [min(wn)]
    if len(wp) > 1:
        wp = [min(wp)]
    return wn[0], wp[0]
