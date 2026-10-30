param(
    [double]$ClockPeriodNs=2.5,
    [string]$ArrayRunTag='20261025_214433'
)
$ErrorActionPreference='Stop'
$culture=[System.Globalization.CultureInfo]::InvariantCulture
$periodArg=$ClockPeriodNs.ToString('0.000',$culture)
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
$outDir="build\supra2_block_ddr_system_ooc\components_$runTag"

function Invoke-VivadoStage([string]$Stage,[string]$Tcl,[string]$LogBase,[string[]]$TclArgs) {
    $stdout=Join-Path $repo "build\${LogBase}.log"
    $journal="build/${LogBase}.jou"
    $internal="build/${LogBase}_internal.log"
    & $vivado -mode batch -journal $journal -log $internal `
        -source $Tcl -tclargs $TclArgs *> $stdout
    if($LASTEXITCODE -ne 0){
        Get-Content -LiteralPath $stdout -Tail 100 -ErrorAction SilentlyContinue
        throw "Vivado $Stage failed; see $stdout"
    }
}

Invoke-VivadoStage 'block-top hierarchical synthesis' `
    'scripts/synth_supra2_block_top_component.tcl' `
    "supra2_sys_${runTag}_block_top_synth" @($runTag,$periodArg)
if(-not (Test-Path "$outDir\block_top_synth.dcp")){throw 'Block-top hierarchical checkpoint is missing.'}

Invoke-VivadoStage 'block-top routed shared-array checkpoint integration' `
    'scripts/integrate_supra2_block_top_array.tcl' `
    "supra2_sys_${runTag}_block_top_array_import" @($runTag,$periodArg,$ArrayRunTag)
if(-not (Test-Path "$outDir\block_top_synth.dcp")){throw 'Block-top checkpoint with integrated array is missing.'}

Invoke-VivadoStage '512-bit DMA synthesis' `
    'scripts/synth_supra2_block_dma_component.tcl' `
    "supra2_sys_${runTag}_dma_synth" @($runTag)
if(-not (Test-Path "$outDir\block_dma_synth.dcp")){throw '512-bit DMA checkpoint is missing.'}

Invoke-VivadoStage '512-bit AXI-to-MIG bridge synthesis' `
    'scripts/synth_axi_to_mig_bridge_component.tcl' `
    "supra2_sys_${runTag}_bridge_synth" @($runTag)
if(-not (Test-Path "$outDir\axi_to_mig_bridge_synth.dcp")){throw 'AXI-to-MIG bridge checkpoint is missing.'}

Invoke-VivadoStage 'complete system shell synthesis and checkpoint assembly' `
    'scripts/synth_supra2_block_ddr_system_shell.tcl' `
    "supra2_sys_${runTag}_shell_synth" @($periodArg,$runTag)
if(-not (Test-Path "$outDir\system_synth.dcp")){throw 'Complete-system synthesized checkpoint is missing.'}

Invoke-VivadoStage 'complete system placement and routing' `
    'scripts/impl_supra2_block_ddr_system_shell.tcl' `
    "supra2_sys_${runTag}_shell_impl" @($periodArg,$runTag)
$timing="$outDir\system_routed_timing.rpt"
$drc="$outDir\system_routed_drc.rpt"
if(-not (Test-Path "$outDir\system_routed.dcp") -or -not (Test-Path $timing) -or -not (Test-Path $drc)){
    throw "Complete-system routed checkpoint/reports are missing; see $outDir"
}
Get-Content -LiteralPath $timing -TotalCount 100
if(-not (Select-String -LiteralPath $timing -Pattern '^All user specified timing constraints are met\.$' -Quiet)){
    throw "Complete system route did not meet ${periodArg} ns; see $timing"
}
"Complete block + 512-bit DMA + AXI-to-MIG bridge implementation PASS at ${periodArg} ns; artifacts: $outDir"
