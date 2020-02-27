"""This module contains layout generators for samplers."""

from typing import Any, Dict

from pybag.enum import MinLenMode, RoundMode, Orientation
from pybag.core import Transform

from bag.util.immutable import Param
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType, MOSPortType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase
from xbase.layout.mos.top import MOSBaseWrapper
from xbase.layout.cap.core import MOMCapCore


class SingleEndedNSampCore(MOSBase):
    """Single ended nMOS sampler core (without capacitor)"""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dict='List of segments for different devices.',
            w_n='nmos width.',
            show_pins='True to show pins.'
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            w_n=0,
            show_pins=False,
        )

    def draw_layout(self) -> None:
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        seg_dict: Dict[str, int] = self.params['seg_dict']
        w_n: int = self.params['w_n']
        show_pins: bool = self.params['show_pins']

        seg_sw1 = seg_dict['sw1']
        seg_sw2 = seg_dict['sw2']
        seg_sw3 = seg_dict['sw3']
        seg_dum = seg_dict['dum']
        seg_sep = self.min_sep_col
        for seg in {seg_sw1, seg_sw2, seg_sw3, seg_dum}:
            if seg % 2 == 1:
                raise ValueError('This layout generator does not support odd number of fingers.')

        # calculate total number of segments based on odd/even
        ldum_col = 0
        sw1_col = ldum_col + seg_dum
        vin_port, cap1_port = MOSPortType.S, MOSPortType.D
        sw2_col = sw1_col + seg_sw1 + 1 - (seg_sw1 % 2)
        cap2_port, vout2_port = vin_port, cap1_port
        sw3_col = sw2_col + seg_sw2 + 1 - (seg_sw2 % 2)
        vout3_port, ref_port = cap2_port, vout2_port
        rdum_col = sw3_col + seg_sw3
        seg_tot = rdum_col + seg_dum
        self.set_mos_size(seg_tot)

        # --- Placement --- #
        sw1 = self.add_mos(0, sw1_col, seg_sw1, w=w_n)
        sw2 = self.add_mos(0, sw2_col, seg_sw2, w=w_n)
        sw3 = self.add_mos(0, sw3_col, seg_sw3, w=w_n)

        # --- Routing --- #
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1

        # 1. input signal
        vin_tid = self.get_track_id(0, MOSWireType.DS_GATE, wire_name='sig', wire_idx=0)
        vin = self.connect_to_tracks([sw1[vin_port]], vin_tid)

        # 2. cap port
        cap_tid = self.get_track_id(0, MOSWireType.DS_GATE, wire_name='sig', wire_idx=1)
        cap = self.connect_to_tracks([sw1[cap1_port], sw2[cap2_port]], cap_tid)

        # 3. output signal
        vout = self.connect_to_tracks([sw2[vout2_port], sw3[vout3_port]], vin_tid)

        # 4. ref signal
        ref_tid = self.get_track_id(0, MOSWireType.DS_GATE, wire_name='sig', wire_idx=2)
        ref = self.connect_to_tracks([sw3[ref_port]], ref_tid)

        # 5. clk, clkb
        clk_tid = self.get_track_id(0, MOSWireType.G, wire_name='clk', wire_idx=0)
        clk = self.connect_to_tracks([sw1.g, sw3.g], clk_tid)
        clkb_tid = self.get_track_id(0, MOSWireType.G, wire_name='clk', wire_idx=1)
        clkb = self.connect_to_tracks([sw2.g], clkb_tid)

        # --- Pins --- #
        self.add_pin('IN', vin, show=show_pins)
        self.add_pin('OUT', vout, show=show_pins)
        self.add_pin('CAP', cap, show=show_pins)
        self.add_pin('REF', ref, show=show_pins)
        self.add_pin('CLK', clk, show=show_pins)
        self.add_pin('CLKB', clkb, show=show_pins)

        # set properties
        self.sch_params = dict(
            seg_dict=dict(
                sw1=seg_sw1,
                sw2=seg_sw2,
                sw3=seg_sw3,
            ),
            lch=pinfo.lch,
            w=pinfo.get_row_place_info(0).row_info.width,
            th=pinfo.get_row_place_info(0).row_info.threshold,
            dum_info=None,
        )


class SingleEndedNSampWrapper(MOSBaseWrapper):
    """Single ended nMOS sampler with edge wrapping."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBaseWrapper.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sampler_params='Params for sampler',
            show_pins='True to show pins',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return SingleEndedNSampCore.get_default_param_values()

    def draw_layout(self) -> None:
        show_pins: bool = self.params['show_pins']
        core_params: Param = self.params['sampler_params']

        core_params = core_params.copy(append=dict(show_pins=False))
        master = self.new_template(SingleEndedNSampCore, params=core_params)

        inst = self.draw_boundaries(master, master.top_layer)

        # re-export pins
        for name in inst.port_names_iter():
            self.reexport(inst.get_port(name), show=show_pins)

        # set properties
        self.sch_params = master.sch_params


class SingleEndedNSampWithCap(MOSBase):
    """Single ended nMOS sampler core with capacitors."""

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            sampler_params='Params for sampler',
            cap_samp_params='Params for sampler MOMCap',
            show_pins='True to show pins',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return SingleEndedNSampCore.get_default_param_values()

    def draw_layout(self) -> None:
        ...
