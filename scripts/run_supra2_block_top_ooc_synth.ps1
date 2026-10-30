param(
    [double]$ClockPeriodNs = 2.5
)

$ErrorActionPreference = 'Stop'
$culture = [System.Globalization.CultureInfo]::InvariantCulture
$periodArg = $ClockPeriodNs.ToString('0.000', $culture)
$periodTag = $periodArg.Replace('.', 'p')
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$repo = 'D:\graduation_design\npu-main\supra2_img_fpga'
$vivado = 'D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
Set-Location $repo
$log = Join-Path $repo "build\supra2_block_top_ooc_${periodTag}ns_${runTag}_vivado.log"
$journal = "build/supra2_block_top_ooc_${periodTag}ns_${runTag}.jou"
$vivadoLog = "build/supra2_block_top_ooc_${periodTag}ns_${runTag}_internal.log"
& $vivado -mode batch -journal $journal -log $vivadoLog `
    -source scripts/synth_supra2_block_top_ooc.tcl `
    -tclargs $periodArg $runTag *> $log
$outDir = "build\supra2_block_top_ooc\vivado_ooc_${periodTag}ns_${runTag}"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $outDir 'synthesized.dcp'))) {
    Get-Content -LiteralPath $log -Tail 100
    throw "Block-top OOC synthesis did not complete; see $log"
}
Get-Content (Join-Path $outDir 'synth_utilization.rpt') -TotalCount 80
Get-Content (Join-Path $outDir 'synth_timing_summary.rpt') -TotalCount 55
"Block-top OOC synthesis PASS; log: $log"
