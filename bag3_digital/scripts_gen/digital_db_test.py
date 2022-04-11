from pathlib import Path
import numpy as np
import pdb
import sys

from bag.core import BagProject

from bag3_digital.design.digital_db.db import DigitalDB, Characterizer

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

#demo_file = 'data/bag3_digital/specs_wrapper/digital_db_test.yaml'
# demo_file = 'data/bag3_digital/specs_db/digital_db_config.yaml'
# demo_file = 'data/bag3_digital/specs_db/digital_db_config_pch.yaml'
demo_file = 'data/bag3_digital/specs_db/db_config.yaml'

mos_type_list = ['nch', 'pch']
th_list = ['lvt', 'svt']

# -----

print("Creating DigitalDB, should run characterization")
ddb = DigitalDB(bprj, config_fname=demo_file, dut_type='pch', force_sim=False)

print("\nQuerying existing values")
q1_dict = dict(
    w=10,
    lch=36,
    mos_type='pch',
    intent='lvt',
    vdd=0.9,
)
cg1, cd1, r1 = ddb.query(q1_dict, env='tt_25')
print(f'cg = {cg1}, cd = {cd1}, res = {r1}')

print("Querying new values")
q2_dict = dict(
    w=20,
    lch=72,
    mos_type='pch',
    intent='svt',
    vdd=0.9,
)
cg2, cd2, r2 = ddb.query(q2_dict, env='tt_25')
print(f'cg = {cg2}, cd = {cd2}, res = {r2}')

# ------
print('Query multiple items')
q5_dict = [q1_dict, q2_dict]
cg_list, cd_list, res_list = ddb.query(q5_dict, env='tt_25')
print(f'cg = {cg_list}, cd = {cd_list}, res = {res_list}')
assert np.array_equal(cg_list[0], cg1[0])
assert np.array_equal(cg_list[1], cg2[0])
# -----
print("Sweep params to param combs")
sweep_params = dict(
    w=list(range(24, 27)),
    lch=[54, 108],
    mos_type=['pch'],
    intent=['lvt', 'svt'],
    vdd=[0.9],
)
q7_dict = ddb.sweep_to_comb(sweep_params)
print("Sweep combs: ", q7_dict)
cg7, cd7, r7 = ddb.query(q7_dict, env='tt_25')
print(f'cg = {cg7}')
print(f'cd = {cd_list}')
print(f'res = {res_list}')
