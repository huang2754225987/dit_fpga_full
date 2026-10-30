set repo [file normalize [file join [file dirname [info script]] ..]]
set out_dir [file join $repo build supra2_block_ddr_system_ooc blackbox_probe]
file mkdir $out_dir
set_param general.maxThreads 1
read_verilog -sv [file join $repo scripts systolic_array_blackbox.sv]
read_verilog -sv [file join $repo scripts probe_systolic_array_parent.sv]
puts "BLACKBOX_PROBE: parent and black box read complete"
synth_design -top probe_systolic_array_parent -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -no_timing_driven
puts "BLACKBOX_PROBE: synth_design complete"
write_checkpoint -force [file join $out_dir systolic_array_blackbox.dcp]
exit
