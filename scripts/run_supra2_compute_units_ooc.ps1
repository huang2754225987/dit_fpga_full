param(
    [string]$StartAt='softmax',
    [string]$RunTag=''
)
$ErrorActionPreference='Stop'
$repo='D:\graduation_design\npu-main\supra2_img_fpga'
$vivado='D:\soft\vitis\Vivado\2021.1\bin\vivado.bat'
if(-not (Test-Path -LiteralPath $vivado)){throw "Vivado not found: $vivado"}
if(-not $env:PROCESSOR_ARCHITECTURE){$env:PROCESSOR_ARCHITECTURE='AMD64'}
Set-Location $repo
if([string]::IsNullOrWhiteSpace($RunTag)){$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'}else{$runTag=$RunTag}
$units=@(
    @{Top='softmax_engine'; Generic='LANES=32 ELEM_W=16'; Tag='softmax'},
    @{Top='supra2_attention_online_pv_batch'; Generic='TOKENS=256 HEAD_DIM=64 TILE=32 COUNT_W=15 EXTERNAL_ARRAY=1'; Tag='attention_self'},
    @{Top='supra2_attention_online_pv_batch'; Generic='TOKENS=32 HEAD_DIM=64 TILE=32 COUNT_W=15 EXTERNAL_ARRAY=1'; Tag='attention_cross'},
    @{Top='supra2_layernorm_modulate_32lane'; Generic='NONE'; Tag='layernorm_adaln'},
    @{Top='supra2_gelu_tanh_32lane'; Generic='NONE'; Tag='gelu_tanh'},
    @{Top='supra2_residual_gate_32lane'; Generic='NONE'; Tag='residual_gate'},
    @{Top='supra2_hadamard_transform'; Generic='MAX_K=2304 MAX_H=256 LANES=32'; Tag='hadamard'},
    @{Top='supra2_activation_quantizer'; Generic='MAX_K=2304 LANES=32'; Tag='activation_quantizer'},
    @{Top='supra2_qlinear_epilogue'; Generic='LANES=32'; Tag='qlinear_epilogue'},
    @{Top='supra2_cfg_combine_32lane'; Generic='LANES=32 COUNT_W=13'; Tag='cfg_combine'},
    @{Top='supra2_euler_update'; Generic='LANES=32 COUNT_W=13'; Tag='euler_update'},
    @{Top='supra2_fp32_add_rne_pipeline'; Generic='NONE'; Tag='fp32_add_pipeline'}
)

$startFound=$false
$startTagFound=$false
foreach($unit in $units){
    if(-not $startFound){
        if($unit.Tag -eq $StartAt){$startFound=$true;$startTagFound=$true}else{continue}
    }
    $base="supra2_compute_${runTag}_$($unit.Tag)"
    $stdout=Join-Path $repo "build\${base}.log"
    $journal="build/${base}.jou"
    $internal="build/${base}_internal.log"
    $start=Get-Date
    $genericHex=[System.BitConverter]::ToString([System.Text.Encoding]::UTF8.GetBytes($unit.Generic)).Replace('-','')
    $tclArgs=@($unit.Top,$genericHex,$runTag,$unit.Tag)
    & $vivado -mode batch -journal $journal -log $internal `
        -source scripts/synth_supra2_compute_unit_ooc.tcl -tclargs $tclArgs *> $stdout
    $exitCode=$LASTEXITCODE
    $elapsed=[math]::Round(((Get-Date)-$start).TotalSeconds,1)
    if($exitCode -ne 0){
        Get-Content -LiteralPath $stdout -Tail 60 -ErrorAction SilentlyContinue
        throw "OOC synthesis failed for $($unit.Top); see $stdout"
    }
    $unitDir="build\supra2_block_ddr_system_ooc\compute_${runTag}\$($unit.Tag)"
    if(-not (Test-Path "$unitDir\synthesized.dcp") -or -not (Test-Path "$unitDir\utilization.rpt")){
        throw "Checkpoint or utilization report missing for $($unit.Top); see $unitDir"
    }
    "PASS $($unit.Tag): ${elapsed}s; report: $unitDir\utilization.rpt"
}
if(-not $startTagFound){throw "Unknown StartAt unit tag: $StartAt"}
"Compute-unit OOC synthesis complete; artifacts: build\supra2_block_ddr_system_ooc\compute_${runTag}"
