# aib_ams_skywater130_workspace 

## BWRC Edition 

open-source release of skywater aib_ams workspace.

## Licensing

This library is licensed under the Apache-2.0 license.  See [here](LICENSE) for full text of the
Apache license.

# Setting up

1. Clone the repository.

2. In the workspace, update submodules:

```
git submodule update --init --recursive
```

3. Change various environment files in `skywater130/workspace_setup` to match your settings.
This will include the paths to tools found in the .bashrc file. 

A shortcut script `setup_script.sh` aids this process on the BWRC machines. 

`bash setup_script.sh`

This will also create pointers to the [OpenAccess](https://en.wikipedia.org/wiki/OpenAccess) and other BAG-related binaries. 

4. Setup environment:
```
source .bashrc
``` 

5. Create cds.lib file:
```
echo 'INCLUDE $BAG_WORK_DIR/cds.lib.core' > cds.lib
```


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

An tristate-inverter, NAND2, NOR2 example are also provided, just change the YAML file path.  Look
at the files in the `data/bag3_digital/specs_gen` for the names.
