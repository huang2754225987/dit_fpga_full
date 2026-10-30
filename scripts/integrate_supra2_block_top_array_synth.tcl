set repo [file normalize [file join [file dirname [info script]] ..]]
set run_tag [lindex $argv 0]
set period_ns [lindex $argv 1]
set array_run_tag [lindex $argv 2]
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
open_checkpoint [file join $out_dir block_top_synth.dcp]
set array_cell [get_cells -hier -quiet -filter {ORIG_REF_NAME == systolic_array || REF_NAME == systolic_array}]
if {[llength $array_cell] != 1} {
    error "Expected exactly one systolic_array cell, found [llength $array_cell]: $array_cell"
}
read_checkpoint -cell $array_cell [file join $repo build supra2_block_ddr_system_ooc "shared_array_${array_run_tag}" synthesized.dcp]
create_clock -name block_clk -period $period_ns [get_ports clk]
write_checkpoint -force [file join $out_dir block_top_synth.dcp]
report_utilization -file [file join $out_dir block_top_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 [file join $out_dir block_top_timing_summary.rpt]
exit
