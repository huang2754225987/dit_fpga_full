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
set out_dir [file join $repo build supra2_dit_runtime_dma "vivado_ooc_${period_tag}ns_${run_tag}"]
file mkdir $out_dir

read_verilog -sv [file join $repo build supra2_dit_ddr_image supra2_dit_ddr_layout.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_axi_read_dma.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_dit_runtime_controller.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_dicache_online_policy.sv]
read_verilog -sv [file join $repo rtl supra2 supra2_dit_runtime_dma.sv]
synth_design -top supra2_dit_runtime_dma -part xczu7ev-ffvc1156-2-e \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized
create_clock -name dma_clk -period $clock_period_ns [get_ports clk]
report_utilization -file [file join $out_dir synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $out_dir synth_timing_summary.rpt]
opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
exit
