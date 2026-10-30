set repo [file normalize [file join [file dirname [info script]] ..]]
set period_ns [lindex $argv 0]
set run_tag [lindex $argv 1]
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 4
read_verilog -sv [file join $repo scripts system_component_blackboxes.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_block_ddr_system_top.sv]
set clock_file [file join $out_dir clock.xdc]
set clock_stream [open $clock_file w]
puts $clock_stream [format {create_clock -name system_clk -period %s [get_ports clk]} $period_ns]
close $clock_stream
synth_design -top supra2_block_ddr_system_top -part xczu7ev-ffvc1156-2-e \
    -generic {AXI_DATA_W=512} -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
foreach {ref_name dcp_name} {
    supra2_block_top block_top_synth.dcp
    supra2_block_dma block_dma_synth.dcp
    axi_to_mig_bridge axi_to_mig_bridge_synth.dcp
} {
    set cell [get_cells -hier -quiet -filter "REF_NAME == $ref_name"]
    if {[llength $cell] != 1} { error "Expected one $ref_name cell, got $cell" }
    read_checkpoint -cell $cell [file join $out_dir $dcp_name]
}
write_checkpoint -force [file join $out_dir system_synth.dcp]
report_utilization -file [file join $out_dir system_synth_utilization.rpt]
exit
