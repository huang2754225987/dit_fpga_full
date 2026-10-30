set repo [file normalize [file join [file dirname [info script]] ..]]
set period_ns 2.500
if {[llength $argv] > 0} { set period_ns [lindex $argv 0] }
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
set out_dir [file join $repo build supra2_block_ddr_system_ooc "shared_array_${run_tag}"]
file mkdir $out_dir
set_param general.maxThreads 4
foreach source [list \
    rtl/pkg/isa_pkg.sv \
    rtl/pkg/fp16_utils_pkg.sv \
    rtl/gemm/mac_int8.sv \
    rtl/gemm/pe.sv \
    rtl/gemm/systolic_array.sv] {
    read_verilog -sv [file join $repo $source]
}
synth_design -top systolic_array -part xczu7ev-ffvc1156-2-e \
    -generic {M=32 N=32 DATA_W=8 ACC_W=32 UNSIGNED_A=0 RUNTIME_UNSIGNED_A=1 PIPE_STAGE=32} \
    -mode out_of_context -flatten_hierarchy none -directive RuntimeOptimized -no_timing_driven
write_checkpoint -force [file join $out_dir synthesized.dcp]
report_utilization -file [file join $out_dir synth_utilization.rpt]
exit
