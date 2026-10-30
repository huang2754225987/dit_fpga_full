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
$log=Join-Path $repo "build\supra2_pv_ooc_${periodTag}ns_${runTag}_vivado.log"
$journal="build/supra2_pv_ooc_${periodTag}ns_${runTag}.jou"
$vivadoLog="build/supra2_pv_ooc_${periodTag}ns_${runTag}_internal.log"
& $vivado -mode batch -journal $journal -log $vivadoLog `
    -source scripts/synth_supra2_pv_ooc.tcl `
    -tclargs $periodArg $runTag *> $log
$outDir="build\supra2_pv_ooc\vivado_ooc_${periodTag}ns_${runTag}"
if($LASTEXITCODE -ne 0 -or -not (Test-Path "$outDir\synthesized.dcp")){
    Get-Content -LiteralPath $log -Tail 80
    throw "PV controller OOC synthesis failed; see $log"
}
Get-Content -LiteralPath "$outDir\synth_utilization.rpt" -TotalCount 80
Get-Content -LiteralPath "$outDir\synth_timing_summary.rpt" -TotalCount 80
"PV controller OOC synthesis completed; reports: $outDir"
