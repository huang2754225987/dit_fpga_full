param([double]$ClockPeriodNs=2.5)
$ErrorActionPreference='Stop'
$culture=[System.Globalization.CultureInfo]::InvariantCulture
$periodArg=$ClockPeriodNs.ToString('0.000',$culture)
$periodTag=$periodArg.Replace('.','p')
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
$outDir="build\supra2_block_ddr_system_ooc\vivado_ooc_${periodTag}ns_${runTag}"
$synthLog=Join-Path $repo "build\supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_synth.log"
$synthJournal="build/supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_synth.jou"
$synthVivadoLog="build/supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_synth_internal.log"
& $vivado -mode batch -journal $synthJournal -log $synthVivadoLog `
    -source scripts/synth_supra2_block_ddr_system_ooc.tcl `
    -tclargs $periodArg $runTag *> $synthLog
$synthDcp=Join-Path $repo "$outDir\synthesized.dcp"
if($LASTEXITCODE -ne 0 -or -not (Test-Path $synthDcp)){
    Get-Content -LiteralPath $synthLog -Tail 100 -ErrorAction SilentlyContinue
    throw "512-bit block DDR system OOC synthesis did not complete; see $synthLog"
}

$implLog=Join-Path $repo "build\supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_impl.log"
$implJournal="build/supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_impl.jou"
$implVivadoLog="build/supra2_block_ddr_system_ooc_${periodTag}ns_${runTag}_impl_internal.log"
& $vivado -mode batch -journal $implJournal -log $implVivadoLog `
    -source scripts/impl_supra2_block_ddr_system_ooc.tcl `
    -tclargs $periodArg $runTag *> $implLog
$timing=Join-Path $outDir 'routed_timing_summary.rpt'
$drc=Join-Path $outDir 'routed_drc.rpt'
if($LASTEXITCODE -ne 0 -or -not (Test-Path $timing) -or -not (Test-Path $drc)){
    Get-Content -LiteralPath $implLog -Tail 100 -ErrorAction SilentlyContinue
    throw "512-bit block DDR system OOC implementation did not complete; see $implLog"
}
Get-Content -LiteralPath $timing -TotalCount 90
if(-not (Select-String -LiteralPath $timing -Pattern '^All user specified timing constraints are met\.$' -Quiet)){
    throw "OOC route completed, but timing did not meet ${periodArg} ns; see $timing"
}
"512-bit block DDR system OOC place-and-route PASS; synthesis log: $synthLog; implementation log: $implLog"
