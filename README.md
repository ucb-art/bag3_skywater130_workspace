# aib_ams_skywater130_release

open-source release of skywater aib_ams workspace.

# Setting up

1. Clone the repository.

2. In the workspace, update submodules:

```
git submodule update --init --recursive
```

3. Change various environment files in `skywater130/workspace_setup` to match your settings.

4. Setup environment:
```
source .bashrc
```

5. Compile pybag if you have OpenAccess source code:

```
cd BAG_framework/pybag
./run_test.sh
```

or obtain pre-compiled library from Blue Cheetah.


# Running example

1. To generate an inverter:

```
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/inv.yaml
```

2. To generate GDS/netlist file:

```
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/inv.yaml -raw
```

3. To run LVS on the generated GDS/netlist file:

```
./run_bag.sh BAG_framework/run_scripts/gen_cell.py data/bag3_digital/specs_gen/inv.yaml -raw -v
```

An tristate-inverter, NAND2, NOR2 example are also provided, just change the YAML file path.
