set repo [file normalize [file join [file dirname [info script]] ..]]
set top [lindex $argv 0]
set generic_hex [lindex $argv 1]
set generic_args [encoding convertfrom utf-8 [binary format H* $generic_hex]]
set run_tag [lindex $argv 2]
set unit_tag [lindex $argv 3]
if {$generic_args eq "NONE"} { set generic_args "" }
if {$top eq "" || $run_tag eq "" || $unit_tag eq ""} {
    error "Usage: synth_supra2_compute_unit_ooc.tcl <top> <generics-or-empty> <run-tag> <unit-tag>"
}
set out_dir [file join $repo build supra2_block_ddr_system_ooc "compute_${run_tag}" $unit_tag]
file mkdir $out_dir
set_param general.maxThreads 4

set file_list [open [file join $repo rtl supra2 supra2_dit_compute_bridge.f] r]
while {[gets $file_list source] >= 0} {
    set source [string trim $source]
    if {$source eq "" || [string index $source 0] eq "#"} { continue }
    read_verilog -sv [file join $repo $source]
}
close $file_list

if {$generic_args eq ""} {
    synth_design -top $top -part xczu7ev-ffvc1156-2-e \
        -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
} else {
    synth_design -top $top -part xczu7ev-ffvc1156-2-e \
        -generic $generic_args -mode out_of_context -flatten_hierarchy none \
        -directive RuntimeOptimized -no_timing_driven
}
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir utilization.rpt]
exit
