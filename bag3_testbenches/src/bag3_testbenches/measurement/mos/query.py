# SPDX-License-Identifier: BSD-3-Clause AND Apache-2.0
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

"""This package contains query classes for transistor parameters."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union, Tuple, Any, Dict, Sequence

import math
from itertools import islice

import numpy as np
import scipy.optimize as sciopt

from bag.util.immutable import ImmutableList
from bag.simulation.core import DesignSpecs
from bag.simulation.hdf5 import load_sim_data_hdf5
from bag.math.interpolate import interpolate_grid
from bag.math.dfun import VectorDiffFunction

if TYPE_CHECKING:
    from bag.simulation.data import SimData
    from bag.math.dfun import DiffFunction


class MOSDBDiscrete(object):
    """Transistor small signal parameters database with discrete width choices.

    This class provides useful query/optimization methods and ways to store/retrieve
    data.

    Parameters
    ----------
    spec_list : List[str]
        list of specification file locations corresponding to widths.
    interp_method : str
        interpolation method.
    meas_type : str
        transistor characterization measurement type.
    vgs_res : float
        vgs resolution used when computing vgs from vstar.
    width_var : str
        the width variable name.
    """

    _tmp_list: ImmutableList[str] = ImmutableList([])

    def __init__(self, spec_list: List[str], interp_method: str = 'spline',
                 meas_type: str = 'mos_ss', vgs_res: float = 5e-3, width_var: str = 'w') -> None:
        self._tot_env_list: Sequence[str] = self._tmp_list
        self._ss_swp_names: ImmutableList[str] = self._tmp_list
        self._ss_outputs: ImmutableList[str] = self._tmp_list
        self._info_list: List[DesignSpecs] = []
        self._ss_list: List[Dict[str, Dict[str, List[DiffFunction]]]] = []
        self._vgs_res = vgs_res

        w_list = []
        for spec in spec_list:
            dsn_specs = DesignSpecs(spec)

            # error checking
            if 'w' in dsn_specs.swp_var_list:
                raise ValueError('MOSDBDiscrete assumes transistor width is not swept.')

            ss_fun_table: Dict[str, Dict[str, List[DiffFunction]]] = {}
            for dsn_name in dsn_specs.dsn_name_iter():
                meas_dir = dsn_specs.get_data_dir(dsn_name, meas_type)
                ss_data = load_sim_data_hdf5(meas_dir / 'ss_params.hdf5')
                ss_data.open_group('ss')

                if not self._tot_env_list:
                    self._tot_env_list = ss_data.sim_envs
                    self._ss_swp_names = ss_data.sweep_params[1:]

                cur_fun_dict = self._make_ss_functions(ss_data, interp_method)
                if not self._ss_outputs:
                    self._ss_outputs = ImmutableList(sorted(cur_fun_dict.keys()))

                ss_fun_table[dsn_name] = cur_fun_dict

            w_list.append(dsn_specs.first_params[width_var])
            self._info_list.append(dsn_specs)
            self._ss_list.append(ss_fun_table)

        self._env_list: Sequence[str] = self._tot_env_list
        self._width_list: ImmutableList[int] = ImmutableList(w_list)
        self._cur_idx = 0
        self._dsn_params = {'w': self._width_list[0]}

    @classmethod
    def _make_ss_functions(cls, ss_data: SimData, interp_method: str
                           ) -> Dict[str, List[DiffFunction]]:
        scale_list: List[Tuple[float, float]] = []
        for name in islice(ss_data.sweep_params, 1, None):
            cur_xvec = ss_data[name]
            scale_list.append((cur_xvec[0], cur_xvec[1] - cur_xvec[0]))

        idx_iter = range(len(ss_data.sim_envs))
        table = {key: [interpolate_grid(scale_list, ss_data[key][idx, ...], method=interp_method,
                                        extrapolate=True, delta=1e-5) for idx in idx_iter]
                 for key in ss_data.signals}

        # add derived parameters
        cgdl = table['cgd']
        cgsl = table['cgs']
        cgbl = table['cgb']
        cdsl = table['cds']
        cdbl = table['cdb']
        csbl = table['csb']
        gml = table['gm']
        ibiasl = table['ibias']
        table['cgg'] = [cgd + cgs + cgb for (cgd, cgs, cgb) in zip(cgdl, cgsl, cgbl)]
        table['cdd'] = [cgd + cds + cdb for (cgd, cds, cdb) in zip(cgdl, cdsl, cdbl)]
        table['css'] = [cgs + cds + csb for (cgs, cds, csb) in zip(cgsl, cdsl, csbl)]
        table['vstar'] = [2 * ibias / gm for (gm, ibias) in zip(gml, ibiasl)]

        return table

    @property
    def width_list(self) -> ImmutableList[int]:
        """ImmutableList[int]: Returns the list of widths in this database."""
        return self._width_list

    @property
    def env_list(self) -> Sequence[str]:
        """Sequence[str]: The list of simulation environments to consider."""
        return self._env_list

    @env_list.setter
    def env_list(self, new_env_list: Sequence[str]) -> None:
        self._env_list = new_env_list

    @property
    def dsn_params(self) -> ImmutableList[str]:
        """ImmutableList[str]: List of design parameters."""
        return self._info_list[self._cur_idx].swp_var_list

    def get_dsn_param_values(self, var: str) -> List[Any]:
        """Returns a list of valid design parameter values."""
        return self._info_list[self._cur_idx].get_swp_values(var)

    def set_dsn_params(self, **kwargs: Any) -> None:
        """Set the design parameters for which this database will query for."""
        self._dsn_params.update(kwargs)
        self._cur_idx = self._width_list.index(self._dsn_params['w'])

    def _get_dsn_name(self, **kwargs: Any) -> str:
        if kwargs:
            self.set_dsn_params(**kwargs)

        combo_list = tuple(self._dsn_params[var] for var in self.dsn_params)
        dsn_name = self._info_list[self._cur_idx].get_design_name(combo_list)
        if dsn_name not in self._ss_list[self._cur_idx]:
            raise ValueError(f'Unknown design name: {dsn_name}.  Did you set design parameters?')

        return dsn_name

    def get_function_list(self, name: str, **kwargs: Any) -> List[DiffFunction]:
        """Returns a list of functions, one for each simulation environment, for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        **kwargs : Any
            design parameter values.

        Returns
        -------
        output : List[DiffFunction]
            the output vector function.
        """
        dsn_name = self._get_dsn_name(**kwargs)
        master_list = self._ss_list[self._cur_idx][dsn_name][name]
        return [master_list[self._get_env_index(env)] for env in self._env_list]

    def get_function(self, name: str, env: str = '', **kwargs: Any
                     ) -> Union[VectorDiffFunction, DiffFunction]:
        """Returns a function for the given output.

        Parameters
        ----------
        name : str
            name of the function.
        env : str
            if not empty, we will return function for just the given simulation environment.
        **kwargs : Any
            design parameter values.

        Returns
        -------
        output : Union[VectorDiffFunction, DiffFunction]
            the output vector function.
        """
        if not env and len(self._env_list) == 1:
            env = self._env_list[0]

        if not env:
            return VectorDiffFunction(self.get_function_list(name, **kwargs))
        else:
            dsn_name = self._get_dsn_name(**kwargs)
            return self._ss_list[self._cur_idx][dsn_name][name][self._get_env_index(env)]

    def get_fun_sweep_params(self, **kwargs: Any) -> Tuple[ImmutableList[str],
                                                           List[Tuple[float, float]]]:
        """Returns interpolation function sweep parameter names and values.

        Parameters
        ----------
        **kwargs : Any
            design parameter values.

        Returns
        -------
        sweep_params : ImmutableList[str]
            list of parameter names.
        sweep_range : List[Tuple[float, float]]
            list of parameter range
        """
        dsn_name = self._get_dsn_name(**kwargs)
        sample_fun = self._ss_list[self._cur_idx][dsn_name]['gm'][0]

        return self._ss_swp_names, sample_fun.input_ranges

    def get_fun_arg(self, vgs: Optional[float] = None, vds: Optional[float] = None,
                    vbs: float = 0.0, vstar: Optional[float] = None, env: str = ''
                    ) -> np.ndarray:
        """Compute argument for small signal parameter functions for the given bias point.

        Either one of vgs and vstar must be specified.  If vds is not specified, we set vds = vgs.
        If vbs is not specified, we set vbs = 0.

        You can specify vstar only if we only consider one simulation environment.

        Parameters
        ----------
        vgs : Optional[float]
            gate-to-source voltage.  For PMOS this is negative.
        vds : Optional[float]
            drain-to-source voltage.  For PMOS this is negative.
        vbs : float
            body-to-source voltage.  For NMOS this is negative.
        vstar : Optional[float]
            vstar, or 2 * id / gm.  This is always positive.
        env : str
            If not empty, will return results for this simulation environment only.

        Returns
        -------
        arg : np.ndarray
            the argument to pass to small signal parameter functions.
        """
        bias_info = self._get_bias_point_info(vgs=vgs, vds=vds, vbs=vbs, vstar=vstar, env=env)
        return np.array([bias_info[key] for key in self._ss_swp_names])

    def _get_bias_point_info(self, vgs: Optional[float] = None, vds: Optional[float] = None,
                             vbs: float = 0.0, vstar: Optional[float] = None, env: str = ''
                             ) -> Dict[str, float]:
        """Compute bias point dictionary from given specs."""
        if vgs is None:
            if vstar is None:
                raise ValueError('At least one of vgs or vstar must be defined.')
            # check we only have one environment
            if not env:
                if len(self._env_list) > 1:
                    raise ValueError('Cannot compute bias point from vstar if we have more than '
                                     'one simulation environment.')
                env = self._env_list[0]

            # compute vgs from vstar spec
            # first, get vgs bounds
            fun_vstar = self.get_function('vstar', env=env)
            vgs_idx = self.get_fun_arg_index('vgs')
            vgs_min, vgs_max = fun_vstar.get_input_range(vgs_idx)
            if vds is None:
                vds_idx = self.get_fun_arg_index('vds')
                vds_min, vds_max = fun_vstar.get_input_range(vds_idx)
                vgs_min = max(vgs_min, vds_min)
                vgs_max = min(vgs_max, vds_max)

            # define vstar function.  Can do batch input.
            ndim = len(self._ss_swp_names)
            op_dict = dict(vds=vds, vbs=vbs)

            def fzero(vtest):
                vstar_arg = np.zeros([np.size(vtest), ndim])
                for idx, key in enumerate(self._ss_swp_names):
                    if key == 'vgs' or key == 'vds' and op_dict['vds'] is None:
                        vstar_arg[:, idx] = vtest
                    else:
                        vstar_arg[:, idx] = op_dict[key]
                return fun_vstar(vstar_arg) - vstar

            # do a coarse sweep to find maximum and minimum vstar.
            # NOTE: we do a coarse sweep because for some technologies, if we
            # are near or below threshold, vstar actually may not be monotonic
            # function of vgs.
            num_pts = int(math.ceil((vgs_max - vgs_min) / self._vgs_res)) + 1
            vgs_vec = np.linspace(vgs_min, vgs_max, num_pts)
            vstar_diff = fzero(vgs_vec)

            if abs(vgs_max) >= abs(vgs_min):
                # NMOS.  We want to find the last vgs with smaller vstar
                idx1 = num_pts - 1 - np.argmax(vstar_diff[::-1] < 0)
                if vstar_diff[idx1] > 0:
                    raise ValueError('vstar = %.4g unachieveable; min vstar = %.4g' %
                                     (vstar, np.min(vstar_diff + vstar)))
                idx2 = idx1 + 1
                if idx2 >= num_pts or vstar_diff[idx2] < 0:
                    raise ValueError('vstar = %.4g unachieveable; max vstar = %.4g' %
                                     (vstar, np.max(vstar_diff + vstar)))
            else:
                # PMOS, we want to find first vgs with smaller vstar
                idx2 = np.argmax(vstar_diff <= 0)
                if vstar_diff[idx2] > 0:
                    raise ValueError('vstar = %.4g unachieveable; min vstar = %.4g' %
                                     (vstar, np.min(vstar_diff + vstar)))
                idx1 = idx2 - 1
                if idx1 < 0 or vstar_diff[idx1] < 0:
                    raise ValueError('vstar = %.4g unachieveable; max vstar = %.4g' %
                                     (vstar, np.max(vstar_diff + vstar)))

            vgs = sciopt.brentq(fzero, vgs_vec[idx1], vgs_vec[idx2])

        if vds is None:
            # set vds if not specified
            vds = vgs

        return dict(vgs=vgs, vds=vds, vbs=vbs)

    def get_fun_arg_index(self, name: str) -> int:
        """Returns the function input argument index for the given variable

        Parameters
        ----------
        name : str
            one of vgs, vds, or vbs.

        Returns
        -------
        idx : int
            index of the given argument.
        """
        return self._ss_swp_names.index(name)

    def query(self, vgs: Optional[float] = None, vds: Optional[float] = None, vbs: float = 0.0,
              vstar: Optional[float] = None, env: str = '') -> Dict[str, np.ndarray]:
        """Query the database for the values associated with the given parameters.

        Either one of vgs and vstar must be specified.  If vds is not specified, we set vds = vgs.
        If vbs is not specified, we set vbs = 0.

        Parameters
        ----------
        vgs : Optional[float]
            gate-to-source voltage.  For PMOS this is negative.
        vds : Optional[float]
            drain-to-source voltage.  For PMOS this is negative.
        vbs : float
            body-to-source voltage.  For NMOS this is negative.
        vstar : Optional[float]
            vstar, or 2 * id / gm.  This is always positive.
        env : str
            If not empty, will return results for this simulation environment only.

        Returns
        -------
        results : Dict[str, np.ndarray]
            the characterization results.
        """
        bias_info = self._get_bias_point_info(vgs=vgs, vds=vds, vbs=vbs, vstar=vstar, env=env)
        fun_arg = np.array([bias_info[key] for key in self._ss_swp_names])
        results = {name: self.get_function(name, env=env)(fun_arg) for name in self._ss_outputs}

        # add bias point information to result
        results.update(bias_info)
        return results

    def _get_env_index(self, env: str) -> int:
        try:
            return self._env_list.index(env)
        except ValueError:
            raise ValueError(f'environment {env} not found.')


def get_db(mos_type: str, dsn_specs: Dict[str, Any]) -> MOSDBDiscrete:
    mos_specs = dsn_specs[mos_type]

    spec_file = mos_specs['spec_file']
    interp_method = mos_specs.get('interp_method', 'spline')
    sim_env = mos_specs.get('sim_env', 'tt_25')

    db = MOSDBDiscrete([spec_file], interp_method=interp_method)
    db.env_list = [sim_env]
    db.set_dsn_params(intent=mos_specs['intent'])

    return db
