set repo [file normalize [file join [file dirname [info script]] ..]]
set run_tag [lindex $argv 0]
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 4
read_verilog -sv [file join $repo build supra2_block13_ddr_traffic block13_ddr_layout.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_block_dma.sv]
synth_design -top supra2_block_dma -part xczu7ev-ffvc1156-2-e \
    -generic {AXI_DATA_W=512} -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
write_checkpoint -force [file join $out_dir block_dma_synth.dcp]
report_utilization -file [file join $out_dir block_dma_utilization.rpt]
exit
