set repo [file normalize [file join [file dirname [info script]] ..]]
set clock_period_ns 2.500
if {[llength $argv] > 0} {
    set clock_period_ns [lindex $argv 0]
}
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} {
    set run_tag [lindex $argv 1]
}
set period_tag [string map {. p} [format %.3f $clock_period_ns]]
set out_dir [file join $repo build supra2_block_top_ooc "vivado_ooc_${period_tag}ns_${run_tag}"]
file mkdir $out_dir

set file_list [open [file join $repo rtl supra2 supra2_dit_compute_bridge.f] r]
while {[gets $file_list source] >= 0} {
    set source [string trim $source]
    if {$source eq "" || [string index $source 0] eq "#"} {
        continue
    }
    read_verilog -sv [file join $repo $source]
}
close $file_list

synth_design -top supra2_block_top -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized
create_clock -name block_clk -period $clock_period_ns [get_ports clk]
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir synth_timing_summary.rpt]
exit
