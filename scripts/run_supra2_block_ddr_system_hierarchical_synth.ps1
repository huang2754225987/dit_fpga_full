param([double]$ClockPeriodNs=2.5)
$ErrorActionPreference='Stop'
$culture=[System.Globalization.CultureInfo]::InvariantCulture
$periodArg=$ClockPeriodNs.ToString('0.000',$culture)
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if(-not (Test-Path -LiteralPath $vivado)){throw "Vivado not found: $vivado"}
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
$sharedDir="build\supra2_block_ddr_system_ooc\shared_array_$runTag"
$outDir="build\supra2_block_ddr_system_ooc\components_$runTag"

function Invoke-VivadoStage([string]$Stage,[string]$Tcl,[string[]]$TclArgs) {
    $base="supra2_hier_${runTag}_${Stage}"
    $stdout=Join-Path $repo "build\${base}.log"
    $journal="build/${base}.jou"
    $internal="build/${base}_internal.log"
    & $vivado -mode batch -journal $journal -log $internal -source $Tcl -tclargs $TclArgs *> $stdout
    if($LASTEXITCODE -ne 0){
        Get-Content -LiteralPath $stdout -Tail 80 -ErrorAction SilentlyContinue
        throw "Vivado $Stage failed; see $stdout"
    }
}

Invoke-VivadoStage 'array' 'scripts/synth_supra2_shared_array_ooc.tcl' @($periodArg,$runTag)
if(-not (Test-Path "$sharedDir\synthesized.dcp")){throw 'Systolic-array synthesis checkpoint is missing.'}
Invoke-VivadoStage 'block' 'scripts/synth_supra2_block_top_component.tcl' @($runTag,$periodArg)
if(-not (Test-Path "$outDir\block_top_synth.dcp")){throw 'Block-top synthesis checkpoint is missing.'}
Invoke-VivadoStage 'array_import' 'scripts/integrate_supra2_block_top_array_synth.tcl' @($runTag,$periodArg,$runTag)
Invoke-VivadoStage 'dma' 'scripts/synth_supra2_block_dma_component.tcl' @($runTag)
if(-not (Test-Path "$outDir\block_dma_synth.dcp")){throw 'DMA synthesis checkpoint is missing.'}
Invoke-VivadoStage 'bridge' 'scripts/synth_axi_to_mig_bridge_component.tcl' @($runTag)
if(-not (Test-Path "$outDir\axi_to_mig_bridge_synth.dcp")){throw 'AXI-to-MIG bridge synthesis checkpoint is missing.'}
Invoke-VivadoStage 'system' 'scripts/synth_supra2_block_ddr_system_shell.tcl' @($periodArg,$runTag)
if(-not (Test-Path "$outDir\system_synth.dcp")){throw 'Hierarchical system synthesis checkpoint is missing.'}
if(-not (Test-Path "$outDir\system_synth_utilization.rpt")){throw 'Hierarchical system utilization report is missing.'}
Get-Content -LiteralPath "$outDir\system_synth_utilization.rpt" -TotalCount 120
"Hierarchical synthesis PASS; no place-and-route run; artifacts: $outDir"
