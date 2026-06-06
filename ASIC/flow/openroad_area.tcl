set PDK_ROOT /OpenROAD-flow-scripts/flow/platforms/nangate45

read_lef $PDK_ROOT/lef/NangateOpenCellLibrary.tech.lef
read_lef $PDK_ROOT/lef/NangateOpenCellLibrary.macro.lef
read_liberty $PDK_ROOT/lib/NangateOpenCellLibrary_typical.lib

read_verilog build/VecAdd2_synth.v
link_design VecAdd2

initialize_floorplan \
  -utilization 70 \
  -aspect_ratio 1.0 \
  -core_space 5 \
  -site FreePDK45_38x28_10R_NP_162NW_34O

source $PDK_ROOT/make_tracks.tcl
place_pins -hor_layers metal3 -ver_layers metal2

global_placement

puts "\n===== Area Report (Nangate45, 70% target util) ====="
report_design_area
puts "===================================================="
