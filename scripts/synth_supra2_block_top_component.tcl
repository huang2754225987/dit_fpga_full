set repo [file normalize [file join [file dirname [info script]] ..]]
set run_tag [lindex $argv 0]
set period_ns [lindex $argv 1]
if {$period_ns eq ""} { set period_ns 2.500 }
set out_dir [file join $repo build supra2_block_ddr_system_ooc "components_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 4
set file_list [open [file join $repo rtl supra2 supra2_dit_compute_bridge.f] r]
while {[gets $file_list source] >= 0} {
    set source [string trim $source]
    if {$source eq "" || [string index $source 0] eq "#"} { continue }
    if {$source eq "rtl/gemm/systolic_array.sv"} { continue }
    read_verilog -sv [file join $repo $source]
}
close $file_list
read_verilog -sv [file join $repo scripts systolic_array_blackbox.sv]
synth_design -top supra2_block_top -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -directive AreaOptimized_high
create_clock -name block_clk -period $period_ns [get_ports clk]
write_checkpoint -force [file join $out_dir block_top_synth.dcp]
report_utilization -file [file join $out_dir block_top_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir block_top_timing_summary.rpt]
exit
