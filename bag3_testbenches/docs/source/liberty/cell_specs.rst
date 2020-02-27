Cell Specification File Format
##############################

gen_specs_file
  The ``generate_cell()`` yaml file name.

props
  Liberty properties of the cells.  Current supported properties are:

  cell_description
    A string describing this cell.

  pin_opposite
    A list of two-element tuples of all differential pins in this cell.

    For each tuple, the first element is a list of positive pins, and the second element is a list
    of negative pins.  Whenever any pins in the positive pins are toggled, all the pins in the
    positive pins list will be toggle in the same way at the same time, and all the pins in the
    negative pins list will be toggle in the opposite way at the same time.  This allows you to
    easily short differential pins together.

pwr_pins
  A dictionary of power supply pins.  The keys are the pin name, and the value is the voltage
  domain name as defined in the library configuration file

gnd_pins
  Same as ``pwr_pins``, but for ground pins.

cond_defaults
  The simulation condition dictionary for all input pins and/or inout pins.  This dictionary
  specify the logical state (0 or 1) that we should set each pin to when running simulation.

  For bus pins with N bits, you can specify a N-bit integer, and the pin values are assigned from
  left to right.  For example, if pin "foo<3:0>" has a value of "1100", then "foo<3>" corresponds
  to MSB (which is 1).  If "bar<0:3>" has a value of "1100", then "bar<3> corresponds to LSB
  (which is 0).

  For inout pins, if they are not specified, they will be left unconnected.

input_pins
  A list of input pin specifications.  Each pin is described by a dictionary, with the following
  entries:

  name
    The pin name.  For bus pins, use angle brackets.
  pwr_pin
    The associated power pin
  gnd_pin
    The associated ground pin
  dw_rise
    The rising driver waveform name
  dw_fall
    The falling driver waveform name
  reset_val
    Either a 0 or 1.  If defined, then this is a reset pin, and the pin will be set to the
    given logical value at the start of the simulation.

    Currently, a reset pin cannot be a bus pin.
  values
    Only needed for bus pins.  This is a list of pin specification dictionaries for each pin
    in the bus.

    For a bus like ``foo<3:0>``, the dictionary for ``foo<3>`` is stored at index 0,
    the dictionary for ``foo<2>`` is stored at index 1, and so on.
  defaults
    Only needed for bus pins.  This is a dictionary of default values of all pins inside this
    bus.  This overrides the input_pins_defaults dictionary.
  timing_info
    See `Timing Specification Format`_.  Only needed for sequential input pins.

input_pins_defaults
   A dictionary of default property values for input pins.

output_pins
  Similar to input_pins.  The output pin specification dictionary has the following entries:

  name
    The pin name.  For bus pins, use angle brackets.

  pwr_pin
    The associated power pin

  gnd_pin
    The associated ground pin

  max_fanout
    Maximum fanout on this output pin

  func
    The logical function of this pin

  cap_info
    See `Maximum Output Capacitance Specification Format`_.

  timing_info
    See `Timing Specification Format`_.


output_pins_defaults
  Similar to input_pins_defaults

inout_pins:
  Similar to output_pins

inout_pins_defaults:
  Similar to output_pins_defaults

min_fanout:
  Minimum fanout.  Used to set minimum output capacitance of each output pins

input_cap_range_scale
  A scale factor used to compute input capacitance range for each input pin.


Maximum Output Capacitance Specification Format
===============================================
A Dictionary describing how maximum output capacitance should be measured.  It has the following
entries:

flop_type
  If present, that means the maximum output capacitiance data should be taking from a
  pre-characterized flop.
flop_pin
  If flop_type is present, this entries describes the corresponding flop output pin name.

max_cap
  Maximum output capacitance in Farads.  If specified, all other entries are ignored, and we
  will use this number directly.
max_trf
  maximum output transistion time, in seconds.
related
  the input pin that should be toggled to get an output waveform
sense
  The logical relationship between the related pin and this output pin.  Now we support
  ``positive_unate`` and ``negative_unate``.
cond
  Any overrides for cond_defaults dictionary.


Timing Specification Format
===========================
For combinational timing constraints, This is a list of dictionaries of all timing constraints
associated with this pin, and how they should be measured.  Each dictionary in this list has the
following entries:

related
  See cap_info section
sense
  See cap_info section
cond
  See cap_info section

For sequential timing constraints, this is a dictionary with the following entries:

flop_type
  the pre-characterized flop name
flop_pin
  the corresponding flop pin name
related
  name of the clock pin
sdf_cond
  A SDF condition string describing when this timing constraint is active
