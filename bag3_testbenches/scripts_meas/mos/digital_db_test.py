from pathlib import Path
import numpy as np
import pdb
import sys

from bag.core import BagProject
from bag.io.file import read_yaml
from bag.util.immutable import Param

from bag.simulation.hdf5 import save_sim_data_hdf5, load_sim_data_hdf5
from bag.simulation.data import SimData, AnalysisData

from bag3_testbenches.measurement.digital.db import DigitalDB, Characterizer

def _info(etype, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(etype, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(etype, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


sys.excepthook = _info

local_dict = locals()
if 'bprj' not in local_dict:
    print('creating BAG project')
    bprj = BagProject()
else:
    print('loading BAG project')
    bprj = local_dict['bprj']

demo_file = 'data/bag3_digital/specs_wrapper/digital_db_test.yaml'
mos_type_list = ['nch', 'pch']
th_list = ['lvt', 'svt']

# -----

print("Creating DigitalDB")
ddb = DigitalDB(bprj, config_fname=demo_file, mos_type='nch', load_from_file=False)
print("\nQuerying existing values")
q1_dict = dict(
    w=1e-6,
    lch=18e-9,
    mos_type='nch',
    intent='lvt',
    vdd=0.9,
)
q1 = ddb.query(q1_dict, env='tt_25')
print("Result ", q1)
print()

print("Querying new values")
q2_dict = dict(
    w=1.5e-6,
    lch=36e-9,
    mos_type='nch',
    intent='svt',
    vdd=0.9,
)
q2 = ddb.query(q2_dict, env='tt_25')
print("Result ", q2)
print()

# ------
print('Query multiple items')
q5_dict = [q1_dict, q2_dict]
q5 = ddb.query(q5_dict, env='tt_25')
print("Results", q5)
assert q5['cg'][0] == q1['cg'][0]
assert q5['cg'][1] == q2['cg'][0]
print()
# -----
print("Sweep params to param combs")
sweep_params=dict(
    w=list(map(lambda x : x * 1e-7, range(1, 3))),
    lch=[20e-9, 30e-9],
    mos_type=['nch'],
    intent=['lvt', 'svt'],
    vdd=[0.9],
)
q7_dict = ddb.sweep_to_comb(sweep_params)
print("Sweep combs: ", q7_dict)
# Query individually
# for query in q7_dict:
#     print(ddb.query(query, env='tt_25'))
#     # Then do some design work here.
# Alt: query all at once
q7 = ddb.query(q7_dict, env='tt_25')
print("Results: ", q7)
