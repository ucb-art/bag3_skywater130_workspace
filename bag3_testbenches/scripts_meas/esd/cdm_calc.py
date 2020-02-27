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

from typing import Dict, Any

import argparse

from bag.math import float_to_si_string
from bag.io import read_yaml

from bag3_testbenches.measurement.esd.cdm import get_rlc_fit, get_rlc_cal, predict_model


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute CDM model parameters.')
    parser.add_argument('specs', help='CDM specs file name.')
    args = parser.parse_args()
    return args


def run_main() -> None:
    args = parse_options()
    specs = read_yaml(args.specs)

    fit_data: Dict[str, Any] = specs.get('fit', None)

    params = None
    if fit_data is not None:
        params = get_rlc_fit(**fit_data)
    else:
        cal_data: Dict[str, Any] = specs.get('calibrate', None)

        if cal_data is not None:
            params = get_rlc_cal(**cal_data)

        pre_specs = specs['predict']
        params.update(pre_specs)

        test = predict_model(params)
        print('predict:')
        for k, v in test.items():
            print(f'{k} = {float_to_si_string(v)}')

    if params is None:
        raise ValueError('No input specification detected')

    print('solved:')
    for k, v in params.items():
        print(f'{k} = {float_to_si_string(v)}')
    print()


if __name__ == '__main__':
    run_main()
