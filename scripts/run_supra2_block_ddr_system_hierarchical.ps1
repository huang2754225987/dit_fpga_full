param([double]$ClockPeriodNs=2.5)
$ErrorActionPreference='Stop'
$culture=[System.Globalization.CultureInfo]::InvariantCulture
$periodArg=$ClockPeriodNs.ToString('0.000',$culture)
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
$sharedDir="build\supra2_block_ddr_system_ooc\shared_array_$runTag"
$topDir="build\supra2_block_ddr_system_ooc\hierarchical_$runTag"

function Invoke-VivadoStage([string]$Stage,[string]$Tcl,[string]$LogBase) {
    $stdout=Join-Path $repo "build\${LogBase}.log"
    $journal="build/${LogBase}.jou"
    $internal="build/${LogBase}_internal.log"
    & $vivado -mode batch -journal $journal -log $internal `
        -source $Tcl -tclargs $periodArg $runTag *> $stdout
    if($LASTEXITCODE -ne 0){
        Get-Content -LiteralPath $stdout -Tail 80 -ErrorAction SilentlyContinue
        throw "Vivado $Stage failed; see $stdout"
    }
}

Invoke-VivadoStage 'shared-array synthesis' 'scripts/synth_supra2_shared_array_ooc.tcl' `
    "supra2_ddr_hier_${runTag}_array_synth"
if(-not (Test-Path "$sharedDir\synthesized.dcp")){throw 'Shared-array synthesis checkpoint is missing.'}
Invoke-VivadoStage 'shared-array implementation' 'scripts/impl_supra2_shared_array_ooc.tcl' `
    "supra2_ddr_hier_${runTag}_array_impl"
if(-not (Test-Path "$sharedDir\routed.dcp")){throw 'Shared-array routed checkpoint is missing.'}

Invoke-VivadoStage 'full-system hierarchical synthesis' `
    'scripts/synth_supra2_block_ddr_system_hierarchical.tcl' `
    "supra2_ddr_hier_${runTag}_top_synth"
if(-not (Test-Path "$topDir\synthesized.dcp")){throw 'Hierarchical full-system checkpoint is missing.'}
Invoke-VivadoStage 'full-system place and route' `
    'scripts/impl_supra2_block_ddr_system_hierarchical.tcl' `
    "supra2_ddr_hier_${runTag}_top_impl"

$timing="$topDir\routed_timing_summary.rpt"
$drc="$topDir\routed_drc.rpt"
if(-not (Test-Path $timing) -or -not (Test-Path $drc) -or -not (Test-Path "$topDir\routed.dcp")){
    throw "Full-system hierarchical implementation reports/checkpoint are missing; see $topDir"
}
Get-Content -LiteralPath $timing -TotalCount 100
if(-not (Select-String -LiteralPath $timing -Pattern '^All user specified timing constraints are met\.$' -Quiet)){
    throw "Full-system route completed but did not meet ${periodArg} ns; see $timing"
}
"Full-system hierarchical OOC place-and-route PASS at ${periodArg} ns; artifacts: $topDir"
