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

from typing import Dict, Any, List, Union, Tuple, cast, Optional, Mapping

from math import log10, floor, sqrt
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import linregress
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from bag.math import float_to_si_string
from bag.core import BagProject
from bag.io.file import read_yaml, write_yaml, render_yaml
from bag.simulation.core import DesignManager, DesignSpecs
from bag.simulation.data import AnalysisData, SimData
from bag.util.immutable import ImmutableList, Param
from bag.simulation.hdf5 import save_sim_data_hdf5, load_sim_data_hdf5
from bag.design.database import ModuleDB
from bag.layout.template import TemplateDB
from bag.util.immutable import to_immutable, ImmutableType

from pybag.enum import DesignOutput
from bag3_testbenches.measurement.digital.timing import CombLogicTimingTB
import pdb


class DigitalDB:
    """The Digital Characterization DataBase

    This database provides query for extraction of cg, cd, rpu/rpd used in logical-effort-based
    design methodologies.
    For nmos resistance values derived from DELAY_H2L and RISE are not valid, for pmos res values
    derived from DELAY_L2H and FALL are not valid.
    """

    def __init__(self,
                 prj: BagProject,
                 config_fname: str,
                 dut_type: str,
                 force_sim: bool = False) -> None:

        self._prj = prj
        self._config_file = config_fname
        self._config = render_yaml(self._config_file, params={})
        self._suf = 'p' if dut_type == 'pmos' else 'n'
        self._force_sim = force_sim
        # self._state_list = [f'cg{self._suf}', f'cd{self._suf}', 'cloadd', 'cloadg',
        #                     f'res{self._suf}']
        # self._meas_types = ['tr', 'tf', 'tdh2l', 'tdl2h']
        # self._param_names = ['cg', 'cd', 'res']
        # self._sim_envs = self._config['env_list']

        # self._sch_db = ModuleDB(self._prj.tech_info, self._config['impl_lib'], prj=self._prj)
        # self._lay_db = TemplateDB(self._prj.grid, self._config['impl_lib'], prj=self._prj)

        # self.sim_db = {}
        self._char_path = Path(self._config['root_dir']) / f'{self._suf}_char.hdf5'

        self._dut_type = 'pch' if dut_type == 'pch' or dut_type == 'pmos' else 'nch'
        self.characterizer = Characterizer(prj, config_fname, self._dut_type)
        self._char_str_path = Path(self._config['root_dir']) / f'{self._suf}_char.yaml'

        self._lookup_fpath = Path(self._config['root_dir']) / 'db_files' / f'{self._suf}_char.yaml'
        try:
            self._lookup: Mapping[int, Any] = read_yaml(self._lookup_fpath)
        except FileNotFoundError:
            self._lookup = {}
        except Exception:
            raise ValueError('Unknown error')

        self.characterize2(self._config['sweep_params'])

    @property
    def dut_type(self):
        return self._dut_type
    
    def get_mos_type_ind(self, mos_type: str) -> int:
        return int(mos_type == 'nch')

    def get_th_type_ind(self, th_type: str):
        intent_list = self._lookup.get('intent_list', [])
        try:
            return intent_list.index(th_type)
        except ValueError:
            intent_list.append(th_type)
            if 'intent_list' not in self._lookup:
                self._lookup['intent_list'] = intent_list
            return len(th_type) - 1

    def get_sim_env_ind(self, sim_env: str):
        sim_envs = self._lookup.get('sim_envs', [])
        try:
            return sim_envs.index(sim_env)
        except ValueError:
            sim_envs.append(sim_env)
            if 'sim_envs' not in self._lookup:
                self._lookup['sim_envs'] = sim_envs
            return len(sim_envs) - 1

    def convert_value_to_indices(self, dd: Dict[str, Union[int, float, str]]) -> \
            Dict[str, Union[int, float]]:
        """
        Convert Dict[str, Union[int, float, str] to Dict[str, Union[int, float]
        Parameters
        ----------
        dd: Dict[str, Union[int, float, str]]
            The values of mos_type/intent/sim_env are converted to unique indices for hashing and
            also doing this lets us save them in hdf5 format

        Returns
        -------
        dr: Dic[str, Union[int, float]]
            Returned dictionary
        """
        dr = dd.copy()
        for k, v in dd.items():
            if isinstance(v, str):
                if k in 'mos_type':
                    dr[k] = self.get_mos_type_ind(v)
                elif k == 'intent':
                    dr[k] = self.get_th_type_ind(v)
                elif k == 'sim_env':
                    dr[k] = self.get_sim_env_ind(v)
        return dr

    def get_hash_value(self, dd: Dict[str, Union[int, float, str]]):
        # return hash(Param(dd.copy()))  # TODO: fix str hash bug

        dr = self.convert_value_to_indices(dd)
        dr['dut_type'] = int(self._dut_type == 'nch')
        ans = ''
        for k, v in dr.items():
            v = str(v)
            if '.' in v:
                v = v.replace('.', '')
            v = int(v)
            # assume there are 3 or fewer digits
            if len(str(v)) > 3:
                raise RuntimeError(f"{k} had unexpected large value {v}")
            ans += f'{v:03}'
        return int(ans)

    @classmethod
    def sweep_to_comb(cls, sweep_params: Dict[str, Union[Any, List[Any]]]):
        """
        Helper function to compute parameter combinations from sweep params
        """
        def _copy_and_add_to_dict(dd, param, val):
            dd = dd.copy()
            dd[param] = val
            return dd

        result = [{}]
        for param_name, vals in sweep_params.items():
            result = [_copy_and_add_to_dict(dd, param_name, v) for dd in result for v in vals]
        return result

    def characterize2(self, sweep_params):
        """
        Creates new private SimData, creates param_comb from sweep params, and populates SimData
        """

        params_comb = self.sweep_to_comb(sweep_params)
        print("Running characterization ... ")
        for param_dict in params_comb:
            key = self.get_hash_value(param_dict)
            if key not in self._lookup or self._force_sim:
                env = param_dict.pop('sim_env')
                self._get_new_result(param_dict, env, key)

    def query(self, params_dict: Union[Dict[str, Any], List[Dict[str, Any]]],
              env: str = None):
        """
        Query the database, use self.swp_var_list() to get a list of valid kwrds for
        params_dict.
        :param params_dict:
        :param env:
        :return:
        """
        if type(params_dict) != list:
            params_dict = [params_dict]

        # TODO: Check validity of input
        # # check that we have all the keys
        # for i, query_dict in enumerate(params_dict):
        #     query_dict = query_dict.copy()
        #     for key in self._info.swp_var_list:
        #         if key not in query_dict.keys():
        #             raise RuntimeError(f'{key} not provided in params_dict')
        #         if key == 'mos_type':
        #             query_dict[key] = self.get_mos_type_ind(query_dict[key])
        #         if key == 'intent':
        #             query_dict[key] = self.get_th_type_ind(query_dict[key])
        #     params_dict[i] = query_dict

        cg, cd, res = [], [], []
        for query_dict in params_dict:
            q = query_dict.copy()
            q['sim_env'] = env
            key = self.get_hash_value(q)
            file = self._lookup.get(key, '')
            if file:
                char_data = load_sim_data_hdf5(Path(file))
            else:
                print("No entry found for given params. Simulating...")
                char_data = self._get_new_result(query_dict, env, key)

            # TODO: Support rise-fall cap query
            cg.append(char_data['cg'])
            cd.append(char_data['cd'])
            res.append(char_data['res'])

        return cg, cd, res

    def _get_new_result(self, params_dict: Dict[str, Any], env: str,
                        hash_key: int = None) -> SimData:
        """
        Update sim_data with new parameters
        Parameters
        ----------
        params_dict: Dict[str, Any]
            parameter dictionary, should not include sim_env. sim_env should be passed separately
            to env parameter
        env:
            simulation environment, also known as sim_env
        hash_key:
            hash key of params_dict, if it is None, it will be computed using get_hash_value()

        Returns
        -------
            SimData containing cg, cd, res values.
        """
        if not hash_key:
            hash_params = params_dict.copy()
            hash_params['sim_env'] = env
            hash_key = self.get_hash_value(hash_params)

        # characterizer expects sim_env to be a separate parameter
        results = self.characterizer.get_char_point(params_dict, hash_key, sim_envs=[env])
        ind_params = self.convert_value_to_indices(params_dict)
        sweep_params = {}
        for k, v in ind_params.items():
            sweep_params[k] = np.array([v])
        results.update(sweep_params)

        ad = AnalysisData(list(sweep_params.keys()), results, is_md=False)
        sim_data = SimData(
            sim_envs=[env],
            data=dict(digital=ad)
        )

        # Save values back
        fpath = self.get_fpath(hash_key)
        save_sim_data_hdf5(sim_data, fpath)
        self._lookup[hash_key] = str(fpath.resolve())
        write_yaml(self._lookup_fpath, self._lookup)
        return sim_data

    def get_fpath(self, hash_value):
        fpath = Path(self._config['root_dir']) / 'db_files' / f'db_{hash_value}.hdf5'
        return fpath

    @staticmethod
    def _get_result(info: DesignSpecs, dsn_name: str) -> Dict[str, Any]:
        """
        Returns the results of a given DesignSpec and the design name
        :param info:
        :param dsn_name:
        :return:
        """
        return read_yaml(info.root_dir / dsn_name / info.summary_fname)

    def _is_sim_data_valid(self, info: DesignSpecs, sim_data: SimData) -> bool:
        """
        Checks whether the loaded sim_data matches the specification file
        :param info:
        :param sim_data:
        :return:
        """
        for dsn_name in info.dsn_name_iter():
            for meas_type in self._meas_types:
                kwrd = f'{dsn_name}_{meas_type}'
                if kwrd not in sim_data.group_list:
                    return False

        return True

    def _get_design_name(self, params_dict: Dict[str, Union[int, float, str]]) -> str:
        """
        Get design name
        :param params_dict:
        :return:
        """
        name = self._info.dsn_basename
        for var, val in params_dict.items():
            if isinstance(val, str) or isinstance(val, int):
                name += f'_{var}_{val}'
            elif isinstance(val, float):
                name += f'_{var}_{float_to_si_string(val)}'
            else:
                raise ValueError('Unsupported parameter type: %s' % (type(val)))

        return name

    def _setup_tb(self, state: str, **kwargs) -> Dict[str, Any]:
        """
        Modify the config parameters for running the DesignManager
        :param state:
        :param kwargs:
        :return:
        """
        print(f'Characterizing state {state} ...')
        tmp_config = deepcopy(self._config)
        tmp_config['dsn_basename'] = f'{self._config["dsn_basename"]}_{state}'
        tmp_config['summary_fname'] = f'summary_{state}.yaml'
        sch_config = tmp_config['schematic_params']
        sch_config['mode'] = state
        meas_config = tmp_config['measurements'][0]
        meas_config['meas_type'] = state
        meas_config['out_fname'] = f'{state}.yaml'
        tb_config = meas_config['testbenches']
        tb_config[state] = tb_config['tb']
        if state.startswith('res'):
            cload = kwargs['cload']
            tb_config[state]['sim_params']['cload'] = cload
        if 'params' in kwargs:
            tmp_config['sweep_params'].update(kwargs['params'])
        del tb_config['tb']
        return tmp_config

    def _create_c_db(self, mode: str) -> SimData:
        """
        Creates the capacitance database
        :param mode:
        :return:
        """
        suf = 'g' if mode == 'cg' else 'd'
        sim_cload: DesignSpecs = self.sim_db[f'cload{suf}']
        sim_c: DesignSpecs = self.sim_db[f'{mode}{self._suf}']

        c_dict = {}
        sim_data = None
        for c_name, cload_name in zip(sim_c.dsn_name_iter(), sim_cload.dsn_name_iter()):
            c_res = self._get_result(sim_c, c_name)
            cload_res = self._get_result(sim_cload, cload_name)

            c_res = cast(SimData, next(iter(c_res.values()))['data'])
            cload_res = cast(SimData, next(iter(cload_res.values()))['data'])
            cload = cload_res['cload']
            c_values = {}
            if sim_data is None:
                sim_data = c_res

            for key in self._meas_types:
                c_values[key] = []
                delay0 = c_res[key]
                delays = cload_res[key]

                initial_shape = delay0.shape
                delays = delays.reshape((-1, delays.shape[-1]))
                delay0 = delay0.reshape((-1, ))
                for d, d_vec in zip(delay0, delays):
                    s, b, _, _, _ = linregress(cload, d_vec)

                    def f_fit(x):
                        return s * x + b

                    ftime = interp1d(cload, d_vec, kind='cubic')
                    tol = 10 ** (floor(log10(cload[-1])) - 3)
                    try:
                        c_val = brentq(lambda x: ftime(x) - d,  cload[0], cload[-1], xtol=tol)
                    except ValueError:
                        c_val = brentq(lambda x: f_fit(x) - d,  cload[0], cload[-1], xtol=tol)
                    c_values[key].append(c_val)

                c_values[key] = np.array(c_values[key])
                c_values[key] = c_values[key].reshape(initial_shape)
            for swp_var in sim_data.sweep_params:
                if swp_var != 'corner':
                    c_values[swp_var] = sim_data[swp_var]

            ana_data = AnalysisData(sim_data.sweep_params, c_values, is_md=sim_data.is_md)
            c_dict[c_name] = ana_data

        cap_data = SimData(sim_data.sim_envs, data=c_dict)
        return cap_data

    def _create_res_db(self, cd_data: SimData, cload: float) -> SimData:
        """
        Creates the resistance database values using the cd values from previous simulations
        :param cd_data:
        :param cload:
        :return:
        """

        if f'cd{self._suf}' not in self.sim_db:
            raise ValueError('There is no Cd information for running resistor characterization')

        sim_res: DesignSpecs = self.sim_db[f'res{self._suf}']
        sim_cd: DesignSpecs = self.sim_db[f'cd{self._suf}']

        res_dict = {}
        for cd_name, res_name in zip(sim_cd.dsn_name_iter(), sim_res.dsn_name_iter()):
            cd_data.open_group(cd_name)
            res_sim_results = self._get_result(sim_res, res_name)

            res_sim_results = next(iter(res_sim_results.values()))['data']
            res_values = {}
            for key in self._meas_types:
                if key in ['tdh2l', 'tdl2h']:
                    res_values[key] = res_sim_results[key] / (cd_data[key] + cload) / np.log(2)
                elif key in ['tf', 'tr']:
                    res_values[key] = res_sim_results[key] / (cd_data[key] + cload) / np.log(9)
                else:
                    raise RuntimeError(f'invalid keyword {key} in resistor results')

            for swp_var in cd_data.sweep_params:
                if swp_var != 'corner':
                    res_values[swp_var] = cd_data[swp_var]

            ana_data = AnalysisData(cd_data.sweep_params, res_values, is_md=cd_data.is_md)
            res_dict[res_name] = ana_data

        res_data = SimData(cd_data.sim_envs, data=res_dict)
        return res_data


def specs_update_helper(specs: Dict[str, Any], params_comb: Dict[str, Any],
                        hash_code: int, mos_type: str, cap_space: np.ndarray):
    """
    Updates the specs dictionary based on the queried parameters, helps setup the simulation
    Same code needs to be run for each wrapper testbench, so helper function makes sure that they
    are all handled the same way
    Parameters
    ----------
    specs : Dict[str, Any]
        the specification dictionary
    params_comb : Dict[str, Any]
        DUT params
    hash_code: int
        hash of params_comb, provided from DigitalDB
    mos_type: str
        dut type
    cap_space: np.ndarray
        Numpy array containing cap values to sweep

    Returns
    -------

    """

    specs['params'].update(params_comb)

    # create a unique dut cell name
    dut_cell_template = specs['impl_cell']
    dut_cell = dut_cell_template + '_' + str(hash_code)

    # update cell names. currently setup for 2 wrappers
    specs['impl_cell'] = dut_cell
    specs['wrapper_params']['dut_params']['dut_cell'] = dut_cell
    specs['wrapper_params']['dut_params']['dut_type'] = mos_type

    cap_swp_info = dict(
        type='LIST',
        values=cap_space
    )
    swp_info = [('cload', cap_swp_info)]
    specs['tbm_specs']['swp_info'] = swp_info

    return specs


class Characterizer:
    """The Digital Characterizer Class

    This characterizer class uses information provided from a DigitalDB to generate instances,
    setup testbenches, run characterizations, and return information
    Manages TestbenchManager and dut wrapper classes

    Currently setup for digital characterizer
    """
    def __init__(self,
                 bprj: BagProject,
                 config_fname: str,
                 dut_type: str) -> None:
        self._prj = bprj
        self._dut_type = dut_type
        self._config_file = config_fname
        self._config = render_yaml(self._config_file, params={})
        self._cload = 10e-15

        sim_info_dir = Path('data', 'bag3_digital', 'specs_db', 'sim_info')
        self._specs_cg_file = sim_info_dir / 'cg_info.yaml'
        self._specs_cd_file = sim_info_dir / 'cd_info.yaml'
        self._specs_res_file = sim_info_dir / 'res_info.yaml'

    def _run_sim(self, specs: Dict[str, Any]):
        result_path = self._prj.simulate_cell(specs, extract=False)
        if result_path is None:
            raise RuntimeError("No simulation results...")

        result = load_sim_data_hdf5(Path(result_path))
        return result

    @staticmethod
    def _get_cap_helper(result, tbm_specs):
        """ Private helper function for calculating cap based on delay results, and finding the
        intercept of reference path delay and dut path delay"""
        get_delay = CombLogicTimingTB.get_output_delay
        dut_fall, dut_rise = get_delay(result, tbm_specs, 'mid_dut', 'dut_in', out_invert=True)
        ref_fall, ref_rise = get_delay(result, tbm_specs, 'mid_ref', 'cap_ref', out_invert=True)
        cap_space = np.ones_like(dut_fall) * result['cload']

        diff_fall = ref_fall - dut_fall
        diff_rise = ref_rise - dut_fall

        min_fall_arg = np.argmin(abs(diff_fall), axis=-1)
        min_rise_arg = np.argmin(abs(diff_rise), axis=-1)
        n_corner = len(min_fall_arg)
        c_rise = cap_space[range(n_corner), min_rise_arg]
        c_fall = cap_space[range(n_corner), min_fall_arg]
        c_avg = (c_rise + c_fall) / 2
        return c_rise, c_fall, c_avg

    def _get_res_helper(self, result, tbm_specs, cd):
        """ helper function for calculating resistor based on cload, cd and delay results"""
        get_delay = CombLogicTimingTB.get_output_delay
        dut_fall, dut_rise = get_delay(result, tbm_specs, 'dut_in', 'dut_out', out_invert=True)

        cload = tbm_specs['sim_params']['cload']
        res_r = dut_rise / (cload + cd[0])
        res_f = dut_fall / (cload + cd[1])
        if self._dut_type == 'nch':
            if np.any(res_f < 0):
                raise ValueError('for dut_type nch resistor became negative')
            return res_f
        if np.any(res_r < 0):
            raise ValueError('for dut_type pch resistor became negative')
        return res_r

    def _get_cap(self, params_combo,  hash_value, mode='cg'):
        spec_file = self._specs_cg_file if mode == 'cg' else self._specs_cd_file
        # top config file should get jinja rendered with params_combo
        config_specs = render_yaml(self._config_file, params_combo)
        root_dir = Path(config_specs['root_dir']) / f'{mode}_{hash_value}'
        config_specs['root_dir'] = root_dir
        # get the substitution dictionary if it exists
        update_dict = config_specs.get(mode, {})
        # copy params so params_combo does not get affected
        cap_params = deepcopy(params_combo)
        cap_params.update(update_dict)
        # replace and render the cap specific file with replacement dictionaries if any
        # for pointers replacement this should happen before rendering
        cap_specs = render_yaml(spec_file, cap_params)
        # for non-pointer replacement just update the spec_file with the content of the top config
        cap_specs.update(config_specs)
        results = self._run_sim(cap_specs)
        return self._get_cap_helper(results, cap_specs['tbm_specs'])

    def _get_res(self, params_combo, hash_value, cd):
        params_combo['cload'] = self._cload
        # top config file should get jinja rendered with params_combo
        config_specs = render_yaml(self._config_file, params_combo)
        root_dir = Path(config_specs['root_dir']) / f'res_{hash_value}'
        config_specs['root_dir'] = root_dir
        # get the substitution dictionary if it exists
        update_dict = config_specs.get('res', {})
        # copy params so params_combo does not get affected
        res_params = deepcopy(params_combo)
        res_params.update(update_dict)
        # replace and render the res specific file with replacement dictionaries if any
        # for pointers replacement this should happen before rendering
        res_specs = render_yaml(self._specs_res_file, res_params)
        # for non-pointer replacement just update the spec_file with the content of the top config
        res_specs.update(config_specs)
        results = self._run_sim(res_specs)
        return self._get_res_helper(results, res_specs['tbm_specs'], cd)

    def get_char_point(self, params: Dict[str, Any], hash_value: int,
                       sim_envs: List[str]) -> Dict[str, np.ndarray]:
        """
        Simulates characterization point returns data
        """

        params_comb = params.copy()
        params_comb.update(dict(dut_type=self._dut_type, sim_envs=sim_envs))
        cg_r, cg_f, cg = self._get_cap(params_comb, hash_value, 'cg')
        cd_bundle = self._get_cap(params_comb, hash_value, 'cd')
        cd_r, cd_f, cd = cd_bundle
        res = self._get_res(params_comb, hash_value, cd_bundle)

        results = dict(
            cg_r=cg_r,
            cg_f=cg_f,
            cg=cg,
            cd_r=cd_r,
            cd_f=cd_f,
            cd=cd,
            res=res
        )

        return results
