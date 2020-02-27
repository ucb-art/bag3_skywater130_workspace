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

import pprint
import argparse

import numpy as np

import matplotlib.pyplot as plt

from bag3_testbenches.measurement.esd.diode import DiodeDB


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DiodeDB usage example.')
    parser.add_argument('specs', help='Diode characterization specs file name.')
    parser.add_argument('-v', '--vbias', dest='vbias', default=-0.5, type=float,
                        help='query bias voltage.')
    parser.add_argument('-n', '--num', dest='num', default=201, type=int,
                        help='number of points to plot.')
    parser.add_argument('-i', '--interp', dest='interp', default='spline',
                        help='interpolation method.')
    parser.add_argument('--log', dest='logy', action='store_true', default=False,
                        help='plot Y axis on log scale.')
    args = parser.parse_args()
    return args


def run_main() -> None:
    args = parse_options()

    db = DiodeDB(args.specs, args.interp)

    pprint.pprint(db.query(args.vbias))

    fun_names = ['ibias', 'rp', 'rs', 'cd', 'cp', 'rel_err']
    nfun = len(fun_names)
    idx_iter = range(nfun)

    fun_list = [db.get_function_list(name) for name in fun_names]
    vlo, vhi = fun_list[0][0].get_input_range(0)
    xvec = np.linspace(vlo, vhi, args.num, endpoint=True)

    fig = plt.figure()
    logy = args.logy
    for idx in idx_iter:
        ax = fig.add_subplot(nfun, 1, idx + 1)
        ax.set_ylabel(fun_names[idx])
        if idx != nfun - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

        cur_flist = fun_list[idx]
        for fidx, env in enumerate(db.env_list):
            yvec = cur_flist[fidx](xvec)
            if logy:
                ax.semilogy(xvec, yvec, label=env)
            else:
                ax.plot(xvec, yvec, label=env)

        ax.legend()

    plt.show()


if __name__ == '__main__':
    run_main()
