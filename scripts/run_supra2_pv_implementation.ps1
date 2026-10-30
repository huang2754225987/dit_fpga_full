param(
    [double]$ClockPeriodNs=2.5,
    [string]$SynthRunTag='20261024_093613'
)
$ErrorActionPreference='Stop'
$culture=[System.Globalization.CultureInfo]::InvariantCulture
$periodArg=$ClockPeriodNs.ToString('0.000',$culture)
$periodTag=$periodArg.Replace('.','p')
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
$synthDcp=Join-Path $repo "build\supra2_pv_ooc\vivado_ooc_${periodTag}ns_${SynthRunTag}\synthesized.dcp"
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
$log=Join-Path $repo "build\supra2_pv_impl_${periodTag}ns_${runTag}_vivado.log"
$journal="build/supra2_pv_impl_${periodTag}ns_${runTag}.jou"
$vivadoLog="build/supra2_pv_impl_${periodTag}ns_${runTag}_internal.log"
& $vivado -mode batch -journal $journal -log $vivadoLog `
    -source scripts/implement_supra2_pv_ooc.tcl `
    -tclargs $periodArg $runTag $synthDcp *> $log
$outDir="build\supra2_pv_ooc\vivado_impl_${periodTag}ns_${runTag}"
$required=@('routed.dcp','routed_utilization.rpt','routed_timing_summary.rpt','routed_drc.rpt')
if($LASTEXITCODE -ne 0 -or @($required | Where-Object { -not (Test-Path (Join-Path $outDir $_)) }).Count -gt 0){
    Get-Content -LiteralPath $log -Tail 100
    throw "PV controller OOC implementation did not complete; see $log"
}
Get-Content -LiteralPath "$outDir\routed_utilization.rpt" -TotalCount 80
Get-Content -LiteralPath "$outDir\routed_timing_summary.rpt" -TotalCount 80
Get-Content -LiteralPath "$outDir\routed_drc.rpt" -TotalCount 80
"PV controller OOC placement and routing completed; reports: $outDir"
