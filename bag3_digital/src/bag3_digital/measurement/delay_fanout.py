"""This package contains class for measuring delay vs fanout"""

from typing import Dict, Any, Tuple, Optional, List, Mapping, Union, Type, cast

from pathlib import Path
from math import ceil, floor
import numpy as np
from scipy.optimize import curve_fit
from copy import deepcopy
import pprint
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from bag.math import float_to_si_string
from bag.design.database import ModuleDB
from bag.layout.template import TemplateDB
from bag.util.importlib import import_class
from bag.util.search import BinaryIterator
from bag.simulation.hdf5 import load_sim_data_hdf5
from bag.core import BagProject
from bag.io.file import read_yaml

from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB


class DelayFanoutMeasurement:
    def __init__(self, bprj: BagProject, spec_file: str = '',
                 spec_dict: Optional[Dict[str, Any]] = None,
                 sch_db: Optional[ModuleDB] = None, lay_db: Optional[TemplateDB] = None) -> None:

        if spec_dict:
            self._specs = spec_dict
        else:
            self._specs = read_yaml(spec_file)

        self._params = self._specs['params']
        self._tb_params = self._specs['tb_params']
        self._tbm_specs = self._specs['tbm_specs']
        self._tbm_class = self._specs['tbm_class']
        self._tbm_print = self._specs['tbm_print']
        self._root_dir = Path(self._specs['root_dir']).resolve()
        self._lay_class = self._specs.get('lay_class', '')
        self._sch_class = self._specs.get('sch_class', '')
        self._extract = self._specs['extract']

        self._prj = bprj

        if sch_db is None:
            self._sch_db = ModuleDB(bprj.tech_info, self._specs['impl_lib'], prj=bprj)
        else:
            self._sch_db = sch_db

        if lay_db is None:
            self._lay_db = TemplateDB(bprj.grid, self._specs['impl_lib'], prj=bprj)
        else:
            self._lay_db = lay_db

    @property
    def lay_class(self) -> str:
        return self._lay_class

    @property
    def sch_class(self) -> str:
        return self._sch_class

    def measure_and_plot(self) -> None:
        fanout_list = np.array(self._specs['fanout_list'])
        tdr_list0, tdf_list0 = np.empty_like(fanout_list, dtype=float), np.empty_like(fanout_list,
                                                                                      dtype=float)

        tdr_list1, tdf_list1 = np.empty_like(fanout_list, dtype=float), np.empty_like(fanout_list,
                                                                                      dtype=float)

        self._params['close2supply'] = True
        for idx, fanout in enumerate(fanout_list):
            self.update_params(fanout)
            tdr, tdf = self.meas_delay()
            tdr_list0[idx] = tdr[0]
            tdf_list0[idx] = tdf[0]

        # self._params['close2supply'] = False
        # for idx, fanout in enumerate(fanout_list):
        #     self.update_params(fanout)
        #     tdr, tdf = self.meas_delay()
        #     tdr_list1[idx] = tdr[0]
        #     tdf_list1[idx] = tdf[0]

        def f(x, m, c):
            return m * x + c

        popt_r, pcov_r = curve_fit(f, fanout_list, tdr_list0)
        popt_f, pcov_f = curve_fit(f, fanout_list, tdf_list0)

        plt.scatter(fanout_list, tdr_list0 * 1e12, label=f'input rising, slope='
                                                      f'{popt_r[0] * 1e12 :0.3f}')
        plt.scatter(fanout_list, tdf_list0 * 1e12, label=f'input falling, slope='
                                                      f'{popt_f[0] * 1e12 :0.3f}')

        # plt.plot(fanout_list, f(fanout_list, popt_r[0], popt_r[1]) * 1e12, label='fit rise')
        # plt.plot(fanout_list, f(fanout_list, popt_f[0], popt_f[1]) * 1e12, label='fit fall')

        # plt.plot(fanout_list, tdr_list1 * 1e12, label='input falling, close2output')
        # plt.plot(fanout_list, tdf_list1 * 1e12, label='input rising, close2output')
        plt.legend()
        plt.xlabel('Fanout')
        plt.ylabel('Delay (in ps)')
        plt.show()

    def meas_delay(self) -> Tuple[np.ndarray, np.ndarray]:
        hdf5_file = self._prj.simulate_cell(self._specs, extract=self._extract, gen_tb=True,
                                            simulate=True, mismatch=False, raw=True)
        sim_data = load_sim_data_hdf5(Path(hdf5_file).resolve())
        return CombLogicTimingTB.get_output_delay(sim_data, self._tbm_specs, 'mid_0', 'mid_1',
                                                  out_invert=True)

    def update_params(self, fanout: int) -> None:
        pn_ratio = 1
        if self._lay_class:
            self._params['params']['seg_list'] = [1, fanout, fanout ** 2, fanout ** 3]
        else:
            for key, val in self._params.items():
                if key.endswith('params'):
                    for idx, par_dict in enumerate(val):
                        par_dict['seg_p'] = fanout ** idx * pn_ratio
                        par_dict['seg_n'] = fanout ** idx
                    return
