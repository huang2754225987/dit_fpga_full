set repo [file normalize [file join [file dirname [info script]] ..]]
set project_file [file join $repo build zcu106_pl project zcu106_pl.xpr]
set period_ns 2.857
set run_tag [clock format [clock seconds] -format %Y%m%d_%H%M%S]
if {[llength $argv] > 0} {
    if {[string is double -strict [lindex $argv 0]]} {
        set period_ns [lindex $argv 0]
        if {[llength $argv] > 1} { set run_tag [lindex $argv 1] }
    } else {
        set run_tag [lindex $argv 0]
    }
}
set frequency_mhz [expr {round(1000.0 / $period_ns)}]
set out_dir [file join $repo build zcu106_pl_${frequency_mhz}mhz $run_tag]
file mkdir $out_dir

set_param general.maxThreads 4
open_project $project_file
set source_set [get_filesets sources_1]
# This board project's XPR marks RTL files AutoDisabled under automatic
# hierarchy updates when Vivado cannot validate its top during project open.
# Keep the current RTL files in the synthesis set and restore the intended top.
set_property source_mgmt_mode None [current_project]
set_property top supra2_zcu106_pl_top $source_set
foreach file_obj [get_files -all -of_objects $source_set] {
    set file_type [get_property FILE_TYPE $file_obj]
    if {[string match -nocase *verilog* $file_type] ||
        [string match -nocase *vhdl* $file_type]} {
        set_property IS_ENABLED true $file_obj
    }
}
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE AreaOptimized_high [get_runs synth_1]

# Keep the board project's MIG/pin configuration, but synchronize block RTL
# sources from the current source list before synthesis. The checked-in XPR
# predates the synchronous BRAM wrapper and FP32 pipeline modules.
set block_file_list [open [file join $repo rtl supra2 supra2_block_top.f] r]
set present_sources {}
foreach file_obj [get_files -of_objects [get_filesets sources_1]] {
    lappend present_sources [file normalize [get_property NAME $file_obj]]
}
while {[gets $block_file_list source] >= 0} {
    set source [string trim $source]
    if {$source eq "" || [string match "#*" $source] ||
        $source eq "rtl/sim/tb_supra2_block_top.sv"} {
        continue
    }
    set source_path [file normalize [file join $repo $source]]
    if {[lsearch -exact $present_sources $source_path] < 0} {
        puts "ADDING_CURRENT_BLOCK_SOURCE=$source_path"
        add_files -norecurse -fileset sources_1 $source_path
        lappend present_sources $source_path
    }
}
close $block_file_list
update_compile_order -fileset sources_1

foreach run_name {ddr_axi_width_synth_1 synth_1} {
    set run_obj [get_runs -quiet $run_name]
    if {[llength $run_obj] > 0} {
        set prior_status [get_property STATUS $run_obj]
        set needs_reset [expr {
            [string match -nocase {*error*} $prior_status] ||
            [string match -nocase {*running*} $prior_status] ||
            [string match -nocase {*queued*} $prior_status]
        }]
        if {$run_name eq "ddr_axi_width_synth_1" &&
            ![file exists [file join $repo build zcu106_pl axi_width_ip ddr_axi_width_7 ddr_axi_width.dcp]]} {
            set needs_reset 1
        }
        if {$needs_reset} {
            set prior_run_dir [get_property DIRECTORY $run_obj]
            foreach artifact {runme.log vivado.jou vivado.log} {
                set prior_artifact [file join $prior_run_dir $artifact]
                if {[file exists $prior_artifact]} {
                    file copy -force $prior_artifact \
                        [file join $out_dir "previous_${run_name}_${artifact}"]
                }
            }
            reset_run $run_obj
        }
    }
}

# Vivado runs synth_1 with the run directory as its current working directory.
# Stage the RTL's repository-relative $readmemh file after any reset_run, which
# clears the run directory, and before launch_runs starts synthesis.
set synth_run_dir [get_property DIRECTORY [get_runs synth_1]]
set synth_lut_dir [file join $synth_run_dir rtl supra2]
file mkdir $synth_lut_dir
file copy -force \
    [file join $repo rtl supra2 gelu_tanh_lut_fp16.hex] \
    [file join $synth_lut_dir gelu_tanh_lut_fp16.hex]
puts "STAGED_GELU_LUT=[file join $synth_lut_dir gelu_tanh_lut_fp16.hex]"

launch_runs synth_1 -jobs 1
wait_on_run synth_1
set synth_status [get_property STATUS [get_runs synth_1]]
puts "SYNTH_STATUS=$synth_status"
if {![string match -nocase {*complete*} $synth_status] ||
    [string match -nocase {*error*} $synth_status]} {
    error "Synthesis did not complete successfully: $synth_status"
}

open_run synth_1
report_clocks -file [file join $out_dir synth_clocks.rpt]
report_utilization -file [file join $out_dir synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 30 \
    -file [file join $out_dir synth_timing_summary.rpt]

set ui_clocks [get_clocks -quiet -filter {NAME =~ *ui_clk*}]
if {[llength $ui_clocks] != 1} {
    puts "AVAILABLE_CLOCKS=[get_clocks]"
    error "Expected one MIG UI clock to constrain at 350 MHz, found [llength $ui_clocks]"
}
set_property PERIOD $period_ns $ui_clocks
puts "TARGET_CLOCK_MHZ=$frequency_mhz CLOCK=[get_property NAME $ui_clocks] PERIOD_NS=$period_ns"
report_clocks -file [file join $out_dir clocks_${frequency_mhz}mhz.rpt]
report_timing_summary -delay_type max -max_paths 50 \
    -file [file join $out_dir timing_pre_route_${frequency_mhz}mhz.rpt]
write_checkpoint -force [file join $out_dir synthesized_${frequency_mhz}mhz.dcp]

opt_design
place_design -directive Explore
phys_opt_design -directive Explore
route_design -directive Explore
write_checkpoint -force [file join $out_dir routed_${frequency_mhz}mhz.dcp]
report_utilization -file [file join $out_dir routed_utilization.rpt]
report_timing_summary -delay_type max -max_paths 50 \
    -file [file join $out_dir routed_timing_summary.rpt]
report_route_status -file [file join $out_dir route_status.rpt]
report_drc -file [file join $out_dir routed_drc.rpt]
write_bitstream -force [file join $out_dir supra2_zcu106_pl_${frequency_mhz}mhz.bit]
puts "BITSTREAM=[file join $out_dir supra2_zcu106_pl_${frequency_mhz}mhz.bit]"
close_project
exit
