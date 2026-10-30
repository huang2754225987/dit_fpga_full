set repo [file normalize [file join [file dirname [info script]] ..]]
if {[llength $argv] < 1} {
    error "Usage: impl_supra2_zcu106_checkpoint.tcl <synth-run-dir> ?period-ns?"
}
set run_dir [file normalize [lindex $argv 0]]
set period_ns 2.5
if {[llength $argv] > 1} { set period_ns [lindex $argv 1] }
set synth_dir [file join $run_dir synth]
set dcp [file join $synth_dir supra2_zcu106_pl_top.dcp]
set out_dir [file join $run_dir impl_400mhz]
if {![file exists $dcp]} { error "Synthesized checkpoint not found: $dcp" }
file mkdir $out_dir

set_param general.maxThreads 1
create_project -in_memory -part xczu7ev-ffvc1156-2-e
set_property board_part xilinx.com:zcu106:part0:2.6 [current_project]
open_checkpoint $dcp

set mig_cells [get_cells -hier -quiet -filter {REF_NAME == ddr4_pl_axi}]
set width_cells [get_cells -hier -quiet -filter {REF_NAME == ddr_axi_width}]
if {[llength $mig_cells] != 1} {
    error "Expected one black-box ddr4_pl_axi cell, found [llength $mig_cells]"
}
if {[llength $width_cells] != 1} {
    error "Expected one black-box ddr_axi_width cell, found [llength $width_cells]"
}
read_checkpoint -cell [lindex $mig_cells 0] \
    [file join $repo build zcu106_pl ip_axi ddr4_pl_axi_8 ddr4_pl_axi.dcp]
read_checkpoint -cell [lindex $width_cells 0] \
    [file join $repo build zcu106_pl axi_width_ip ddr_axi_width_7 ddr_axi_width.dcp]

set mig_dir [file join $repo build zcu106_pl ip_axi ddr4_pl_axi_8]
read_xdc [file join $mig_dir par ddr4_pl_axi.xdc]
read_xdc [file join $mig_dir ddr4_pl_axi_board.xdc]
read_xdc [file join $repo build zcu106_pl_pins.xdc]

report_clocks -file [file join $out_dir clocks_before_400mhz.rpt]
set ui_clocks [get_clocks -quiet -filter {NAME =~ *ui_clk*}]
if {[llength $ui_clocks] != 1} {
    puts "AVAILABLE_CLOCKS=[get_clocks]"
    error "Expected one MIG UI clock, found [llength $ui_clocks]"
}
set_property PERIOD $period_ns $ui_clocks
puts "IMPLEMENT_CLOCK=[get_property NAME $ui_clocks] PERIOD_NS=$period_ns"
report_clocks -file [file join $out_dir clocks_400mhz.rpt]
report_timing_summary -delay_type max -max_paths 50 \
    -file [file join $out_dir timing_pre_route_400mhz.rpt]
write_checkpoint -force [file join $out_dir synthesized_400mhz.dcp]

opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir routed_400mhz.dcp]
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 50 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_route_status -file [file join $out_dir route_status.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
close_project
exit
