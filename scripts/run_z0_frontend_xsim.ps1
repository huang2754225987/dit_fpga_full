param(
    [string]$VivadoRoot = 'D:\soft\vitis\Vivado\2021.1',
    [string]$Msys2Root = 'D:\C\msys2\ucrt64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    $outputDir = 'build/supra2_x_embed_qlinear'
    python -m compiler.supra2.prepare_x_embed_qlinear_projection --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw 'Archive-z0 x_embed fixture generation failed' }

    & (Join-Path $PSScriptRoot 'run_xsim_standalone.ps1') `
        -FileList 'rtl/supra2/qlinear_projection.f' `
        -Top 'tb_supra2_qlinear_projection' `
        -Snapshot 'supra2_x_embed_qlinear' `
        -Defines @('SUPRA2_XSIM_XEMBED') `
        -PassPattern 'SUPRA2_QPROJ_PASS M=256 N=576 K=16' `
        -VivadoRoot $VivadoRoot -Msys2Root $Msys2Root `
        -TestPlusArgs @(
            'M=256', 'N=576', 'K=16', 'H=16',
            "RAW_INPUT_FILE=$outputDir/raw_input_fp16.hex",
            "EXPECTED_Q_FILE=$outputDir/expected_quantized_int8.hex",
            "EXPECTED_ROTATED_FILE=$outputDir/rotated_fp16.hex",
            "WEIGHT_FILE=$outputDir/weight_int8.hex",
            "ACC_FILE=$outputDir/accumulator_int32.hex",
            "OUTPUT_FILE=$outputDir/output_fp16.hex",
            "ACTUAL_OUTPUT_FILE=$outputDir/rtl_output_fp16.hex",
            "REFERENCE_OUTPUT_FILE=$outputDir/reference_output_fp16.hex",
            "TOKEN_SCALE_FILE=$outputDir/token_scale_fp32.hex",
            "WEIGHT_SCALE_FILE=$outputDir/weight_scale_fp32.hex",
            "BIAS_FILE=$outputDir/bias_fp32.hex"
        )

    & (Join-Path $PSScriptRoot 'run_position_add_xsim.ps1') `
        -InputFile "$outputDir/rtl_output_fp16.hex" `
        -VivadoRoot $VivadoRoot -Msys2Root $Msys2Root
    if (-not $?) { throw 'RTL position-add regression failed after x_embed' }

    $normDir = 'build/supra2_z0_layernorm'
    python -m compiler.supra2.prepare_layernorm_modulate `
        --input-file 'build/supra2_position_add/rtl_output_fp16.hex' `
        --output-dir $normDir --block 0
    if ($LASTEXITCODE -ne 0) { throw 'RTL-fed LayerNorm fixture generation failed' }
    & (Join-Path $PSScriptRoot 'run_xsim_standalone.ps1') `
        -FileList 'rtl/supra2/layernorm_modulate.f' `
        -Top 'tb_supra2_layernorm_modulate' `
        -Snapshot 'supra2_z0_layernorm' `
        -PassPattern 'SUPRA2_LAYERNORM_MODULATE_PASS' `
        -VivadoRoot $VivadoRoot -Msys2Root $Msys2Root `
        -TestPlusArgs @(
            "X_FILE=$normDir/x_fp16.hex",
            "SHIFT_FILE=$normDir/shift_fp16.hex",
            "SCALE_FILE=$normDir/scale_fp16.hex",
            "EXPECTED_FILE=$normDir/reference_mod_fp16.hex",
            "QKV_FILE=$normDir/qkv_xraw_fp16.hex",
            "OUTPUT_FILE=$normDir/rtl_modulated_fp16.hex"
        )
    if (-not $?) { throw 'RTL LayerNorm/AdaLN regression failed after position add' }

    Write-Host 'SUPRA2_Z0_FRONTEND_PASS z0 -> x_embed RTL -> position-add RTL -> LayerNorm/AdaLN RTL'
}
finally {
    Pop-Location
}
