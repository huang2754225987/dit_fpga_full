set repo [file normalize [file join [file dirname [info script]] ..]]
set run_tag [lindex $argv 0]
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 4
read_verilog -sv [file join $repo rtl axi_to_mig_bridge.sv]
synth_design -top axi_to_mig_bridge -part xczu7ev-ffvc1156-2-e \
    -generic {AXI_DATA_W=512 AXI_ADDR_W=32 AXI_ID_W=4 MIG_DATA_W=64 MIG_ADDR_W=29 MIG_MASK_W=8 RATIO=8 READ_FIFO_BEATS=8} \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
write_checkpoint -force [file join $out_dir axi_to_mig_bridge_synth.dcp]
report_utilization -file [file join $out_dir axi_to_mig_bridge_utilization.rpt]
exit
