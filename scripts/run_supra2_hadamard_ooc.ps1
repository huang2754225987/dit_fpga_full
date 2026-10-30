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
$log = Join-Path $repo "build\supra2_hadamard_ooc_${periodTag}ns_${runTag}_vivado.log"
$journal = "build/supra2_hadamard_ooc_${periodTag}ns_${runTag}.jou"
$vivadoLog = "build/supra2_hadamard_ooc_${periodTag}ns_${runTag}_internal.log"
& $vivado -mode batch -journal $journal -log $vivadoLog `
    -source scripts/synth_supra2_hadamard_ooc.tcl `
    -tclargs $periodArg $runTag *> $log
$outDir = "build\supra2_hadamard_ooc\vivado_ooc_${periodTag}ns_${runTag}"
$timingReport = Join-Path $outDir 'routed_timing_summary.rpt'
$drcReport = Join-Path $outDir 'routed_drc.rpt'
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $timingReport) -or -not (Test-Path $drcReport)) {
    Get-Content -LiteralPath $log -Tail 100
    throw "Hadamard OOC implementation did not complete; see $log"
}
Get-Content $timingReport -TotalCount 90
if (Select-String -LiteralPath $log -Pattern 'Synth 8-295' -Quiet) {
    throw "Hadamard implementation contains synthesis-inferred timing loops; see $log"
}
if (-not (Select-String -LiteralPath $timingReport -Pattern '^All user specified timing constraints are met\.$' -Quiet)) {
    throw "Hadamard OOC route completed, but timing did not meet ${periodArg} ns; see $timingReport"
}
"Hadamard OOC place-and-route PASS; log: $log"
