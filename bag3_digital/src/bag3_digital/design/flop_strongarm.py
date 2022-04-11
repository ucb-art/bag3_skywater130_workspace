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

from typing import Mapping, Dict, Any, Tuple, Sequence

from copy import deepcopy

from xbase.layout.mos.base import MOSBasePlaceInfo
from xbase.layout.enum import MOSType

from bag3_digital.measurement.cap.delay_match import CapDelayMatch
from bag3_testbenches.design.base import DesignerBase

from bag3_digital.layout.stdcells.util import STDCellWrapper
from bag3_digital.layout.sampler.flop_strongarm import FlopStrongArm


class FlopStrongArmDesigner(DesignerBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        DesignerBase.__init__(self, *args, **kwargs)

    async def async_design(self, cload: float, freq: float, vdd: float,
                           trf_in: float, sim_envs: Sequence[str], pinfo: Mapping[str, Any],
                           cap_mm_params: Dict[str, Any],
                           strongarm_tile_name: str = '', sr_tile_name: str = '',
                           ridx_p: int = -1, ridx_n: int = 0,
                           **kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """ Design a flop strongarm. """
        _, tinfo_table = MOSBasePlaceInfo.make_place_info(self.grid, pinfo)

        sr_pinfo: MOSBasePlaceInfo = tinfo_table[sr_tile_name]
        sa_pinfo: MOSBasePlaceInfo = tinfo_table[strongarm_tile_name]

        sa_row_info = self._get_row_info(sa_pinfo)
        sr_row_info = self._get_row_info(sr_pinfo)

        seg_min = 1

        sa_params = self._design_strongarm_frontend(seg_min=seg_min, **sa_row_info, **kwargs)
        sr_params = self._design_sr_latch(seg_min=seg_min, **sr_row_info, **kwargs)

        subblock_params = dict(
            sa_params=sa_params,
            sr_params=sr_params
        )
        for opt_param in ['has_rstlb', 'swap_outbuf']:
            if opt_param in kwargs:
                subblock_params[opt_param] = kwargs[opt_param]

        dut_params = dict(
            pinfo=self._get_pinfo_subdict(pinfo, [strongarm_tile_name, sr_tile_name]),
            **subblock_params
        )

        mm_params = self._get_cap_delay_match_mm_params(cap_mm_params, vdd, freq,
                                                        trf_in, sa_row_info)

        designed_perf = await self._simulate(dut_params, mm_params)

        return subblock_params, designed_perf

    async def _simulate(self, dut_params: Mapping[str, Any],
                        mm_params: Mapping[str, Any]) -> Dict[str, Any]:
        """
        """
        designed_perf = {}

        gen_params = dict(
            cls_name=FlopStrongArm.get_qualified_name(),
            draw_taps=True,
            params=dut_params,
        )
        designed_dut = await self.async_new_dut('flop_strongarm', STDCellWrapper, gen_params)

        for in_pin in ['inp', 'inn', 'clk']:
            mm = self.make_mm(CapDelayMatch, dict(**mm_params, in_pin=in_pin))
            results = await self.async_simulate_mm_obj(f'flop_strongarm_c_{in_pin}', designed_dut, mm)
            c_rise = results.data['cap_rise']
            c_fall = results.data['cap_fall']
            designed_perf[f'c_{in_pin}'] = 0.5 * (c_rise + c_fall)

        # TODO: currently extract is set to true because LVS is being run (instead of simulation)
        # rcx may not be implemented, hence the NotImplementedError exception is being handled.
        # Once the simulation is implemented, the try-except clause can be removed
        try:
            dut = await self.async_new_dut('flop_strongarm', STDCellWrapper, gen_params, extract=True)
        except NotImplementedError:
            pass
        except Exception as e:
            raise e

        # TODO: run simulation
        return designed_perf

    @staticmethod
    def _design_strongarm_frontend(w_n: int, w_p: int, seg_min: int, **kwargs) -> Dict[str, Any]:
        # FIXME: everything is min size for now
        # Note: layout generator forces the following for the number of segments:
        #   in, nfb, and pfb must be even
        #   tail must be a multiple of 4
        seg_list = ['in', 'tail', 'nfb', 'pfb', 'sw']
        w_list = ['in', 'tail', 'nfb', 'pfb']
        nmos_w_list = ['in', 'tail', 'nfb']

        seg_dict = {k: seg_min for k in seg_list}
        for seg_even in ['in', 'nfb', 'pfb']:
            seg_dict[seg_even] *= 2
        seg_dict['tail'] *= 4

        w_dict = {k: w_n if k in nmos_w_list else w_p for k in w_list}

        params = dict(
            seg_dict=seg_dict,
            w_dict=w_dict,
        )
        if 'has_bridge' in kwargs:
            params['has_bridge'] = kwargs['has_bridge']

        return params

    @staticmethod
    def _design_sr_latch(w_n: int, w_p: int, seg_min: int,
                         has_rstlb: bool = False, **kwargs) -> Dict[str, Any]:
        # FIXME: everything is min size for now
        # Note: layout generator forces the following:
        #   ps, nr, rst, and obuf must have even number of segments
        #   w_nbuf = w_nr, w_pbuf = w_ps

        w_list = ['nfb', 'pfb', 'ps', 'nr']
        w_dict = {k: w_n if k.startswith('n') else w_p for k in w_list}

        seg_list = ['fb', 'ps', 'nr', 'ibuf', 'obuf']
        if has_rstlb:
            seg_list += ['rst']
        seg_dict = {}
        for k in seg_list:
            if k in ['ps', 'nr', 'rst', 'obuf'] and seg_min & 1:
                seg_dict[k] = seg_min + 1
            else:
                seg_dict[k] = seg_min

        # FIXME: why does odd seg result in LVS errors for ibuf?
        for k in ['ibuf']:
            seg_dict[k] = 2

        params = dict(
            seg_dict=seg_dict,
            w_dict=w_dict,
            has_rstb=has_rstlb,
        )
        if 'has_outbuf' in kwargs:
            params['has_outbuf'] = kwargs['has_outbuf']

        return params

    def _get_row_info(self, pinfo: MOSBasePlaceInfo) -> Dict[str, Any]:
        wn, wp = self._get_default_width(pinfo)
        thn, thp = self._get_th(pinfo)
        return dict(
            w_n=wn,
            w_p=wp,
            th_n=thn,
            th_p=thp,
            lch=pinfo.lch
        )

    def _get_pinfo_subdict(self, pinfo_dict: Mapping[str, Any], tile_names: Sequence
                           ) -> Mapping[str, Any]:
        pinfo_dict = deepcopy(pinfo_dict)
        pinfo_tiles = []
        for tile_dict in pinfo_dict['tiles']:
            if tile_dict['name'] in tile_names:
                pinfo_tiles.append(tile_dict)
        pinfo_dict['tiles'] = pinfo_tiles

        tile_specs_place_infos = {}
        for k, v in pinfo_dict['tile_specs']['place_info'].items():
            if k in tile_names:
                tile_specs_place_infos[k] = v
        pinfo_dict['tile_specs']['place_info'] = tile_specs_place_infos

        abut_list = []
        for abut in pinfo_dict['tile_specs']['abut_list']:
            if all(map(lambda edge: edge[0] in tile_names, abut['edges'])):
                abut_list.append(abut)
        pinfo_dict['tile_specs']['abut_list'] = abut_list

        return pinfo_dict

    def _get_default_width(self, pinfo: MOSBasePlaceInfo) -> Tuple[int, int]:
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

    def _get_th(self, pinfo: MOSBasePlaceInfo) -> Tuple[str, str]:
        # Assume every row has same threshold for each MOSType
        thn, thp = None, None
        for row_place_info in map(pinfo.get_row_place_info, range(pinfo.num_rows)):
            th = row_place_info.row_info.threshold
            if row_place_info.row_info.row_type is MOSType.nch:
                thn = th
            elif row_place_info.row_info.row_type is MOSType.pch:
                thp = th
            if thn is not None and thp is not None:
                break
        return thn, thp

    def _get_cap_delay_match_mm_params(self, cap_mm_params: Mapping[str, Any], vdd: float,
                                       freq: float, trf_in: float,
                                       sa_row_info: Mapping[str, Any]) -> Dict[str, Any]:
        pins = ['clk', 'inn', 'inp', 'outp', 'outn']
        pwr_domain = {pin: ('VSS', 'VDD') for pin in pins}
        tbm_specs = cap_mm_params['tbm_specs']
        tbm_specs['sim_params'].update(dict(
            t_rst=5 * trf_in,
            t_rst_rf=trf_in,
            t_bit=1.0/freq,
            t_rf=trf_in,
        ))
        tbm_specs['pwr_domain'] = pwr_domain
        tbm_specs['sup_values'] = dict(VDD=vdd, VSS=0.0)

        if 'buf_params' in cap_mm_params and 'inv_params' in cap_mm_params['buf_params']:
            inv_params_list = cap_mm_params['buf_params']['inv_params']
            assert len(inv_params_list) == 2
        else:
            inv_params_list = [{}, {}]
        for inv_params in inv_params_list:
            for var in ['lch', 'w_p', 'w_n', 'th_n', 'th_p']:
                if var not in inv_params:
                    inv_params[var] = sa_row_info[var]

        buf_params = dict(
            inv_params=inv_params_list,
            export_pins=True
        )

        load_list = [dict(pin=pin_name, type='cap', value='c_out') for pin_name in ['outp', 'outn']]

        return dict(
            tbm_specs=tbm_specs,
            buf_params=buf_params,
            search_params=cap_mm_params['search_params'],
            load_list=load_list
        )
