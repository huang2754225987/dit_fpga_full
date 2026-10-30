param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [ValidateRange(0,13)][int]$Block = 0,
    [string]$InputFile = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vlib = Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog = Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim = Join-Path $ModelSimRoot 'win64\vsim.exe'
$license = Join-Path $ModelSimRoot 'LICENSE.TXT'
$normLibrary = 'build/modelsim/work_supra2_norm'
$qprojLibrary = 'build/modelsim/work_supra2_norm_qproj'
$qkLibrary = 'build/modelsim/work_supra2_attention_qk'
$residualLibrary = 'build/modelsim/work_supra2_self_attn_residual'
$tag = if ($Block -eq 0) { 'supra2' } else { "supra2_b$Block" }
$normDir = "build/${tag}_layernorm_modulate"
$qprojDir = "build/${tag}_qkv_after_layernorm"
$qkDir = "build/${tag}_attention_qk"
$pvDir = "build/${tag}_attention_online_pv_batch"
$projDir = "build/${tag}_self_attn_proj"
$residualDir = "build/${tag}_self_attn_residual"

foreach ($tool in @($vlib, $vlog, $vsim, $license)) {
    if (-not (Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license
    $normFixtureArgs = @('--output-dir', $normDir, '--block', "$Block")
    if ($InputFile) { $normFixtureArgs += @('--input-file', $InputFile) }
    python -m compiler.supra2.prepare_layernorm_modulate @normFixtureArgs
    if ($LASTEXITCODE -ne 0) { throw 'LayerNorm fixture generation failed' }

    if (-not (Test-Path $normLibrary)) {
        & $vlib $normLibrary
        if ($LASTEXITCODE -ne 0) { throw 'LayerNorm vlib failed' }
    }
    & $vlog -sv -work $normLibrary -f rtl/supra2/layernorm_modulate.f
    if ($LASTEXITCODE -ne 0) { throw 'LayerNorm RTL compile failed' }
    $normArgs = @(
        '-c', '-work', $normLibrary, 'tb_supra2_layernorm_modulate',
        "+X_FILE=$normDir/x_fp16.hex",
        "+SHIFT_FILE=$normDir/shift_fp16.hex",
        "+SCALE_FILE=$normDir/scale_fp16.hex",
        "+EXPECTED_FILE=$normDir/reference_mod_fp16.hex",
        "+QKV_FILE=$normDir/qkv_xraw_fp16.hex",
        "+OUTPUT_FILE=$normDir/rtl_modulated_fp16.hex",
        '-do', 'run -all; quit -f'
    )
    $normOutput = & $vsim @normArgs 2>&1
    $normOutput | Tee-Object -FilePath "$normDir/modelsim.log"
    $normText = $normOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $normText -notmatch 'SUPRA2_LAYERNORM_MODULATE_PASS' -or
        $normText -match '\*\* Fatal|Errors: [1-9]') {
        throw "LayerNorm/AdaLN RTL regression failed; see $normDir/modelsim.log"
    }

    python -m compiler.supra2.prepare_qlinear_projection `
        --layer "blocks.$Block.self_attn.qkv" --fp16-hadamard-frontend `
        --raw-input-file "$normDir/rtl_modulated_fp16.hex" --output-dir $qprojDir
    if ($LASTEXITCODE -ne 0) { throw 'LayerNorm-derived QKV fixture generation failed' }
    if (-not (Test-Path $qprojLibrary)) {
        & $vlib $qprojLibrary
        if ($LASTEXITCODE -ne 0) { throw 'QLinear vlib failed' }
    }
    & $vlog -sv -work $qprojLibrary -f rtl/supra2/qlinear_projection.f
    if ($LASTEXITCODE -ne 0) { throw 'QLinear RTL compile failed' }
    $qprojArgs = @(
        '-c', '-work', $qprojLibrary, 'tb_supra2_qlinear_projection',
        '+M=256', '+N=1728', '+K=576', '+H=64',
        "+RAW_INPUT_FILE=$qprojDir/raw_input_fp16.hex",
        "+EXPECTED_Q_FILE=$qprojDir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$qprojDir/rotated_fp16.hex",
        "+WEIGHT_FILE=$qprojDir/weight_int8.hex",
        "+ACC_FILE=$qprojDir/accumulator_int32.hex",
        "+OUTPUT_FILE=$qprojDir/output_fp16.hex",
        "+ACTUAL_OUTPUT_FILE=$qprojDir/rtl_output_fp16.hex",
        "+REFERENCE_OUTPUT_FILE=$qprojDir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$qprojDir/token_scale_fp32.hex",
        "+WEIGHT_SCALE_FILE=$qprojDir/weight_scale_fp32.hex",
        "+BIAS_FILE=$qprojDir/bias_fp32.hex",
        '-do', 'run -all; quit -f'
    )
    $qprojOutput = & $vsim @qprojArgs 2>&1
    $qprojOutput | Tee-Object -FilePath "$qprojDir/modelsim.log"
    $qprojText = $qprojOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $qprojText -notmatch 'SUPRA2_QPROJ_PASS M=256 N=1728 K=576' -or
        $qprojText -notmatch 'SUPRA2_QPROJ_FRONTEND_H_MISMATCHES 0' -or
        $qprojText -notmatch 'SUPRA2_QPROJ_FRONTEND_Q_MISMATCHES 0' -or
        $qprojText -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or
        $qprojText -match '\*\* Fatal|Errors: [1-9]') {
        throw "LayerNorm-fed QKV projection failed; see $qprojDir/modelsim.log"
    }

    python -m compiler.supra2.prepare_attention_qk `
        --qkv-file "$qprojDir/rtl_output_fp16.hex" --output-dir $qkDir
    if ($LASTEXITCODE -ne 0) { throw 'RTL QK fixture generation failed' }
    if (-not (Test-Path $qkLibrary)) {
        & $vlib $qkLibrary
        if ($LASTEXITCODE -ne 0) { throw 'QK vlib failed' }
    }
    & $vlog -sv -work $qkLibrary -f rtl/supra2/attention_qk.f
    if ($LASTEXITCODE -ne 0) { throw 'QK RTL compile failed' }
    $qkArgs = @(
        '-c', '-work', $qkLibrary, 'tb_supra2_attention_qk',
        "+Q_FILE=$qkDir/q_fp16.hex",
        "+K_FILE=$qkDir/k_fp16.hex",
        "+EXPECTED_FILE=$qkDir/expected_scores_fp16.hex",
        "+ACTUAL_FILE=$qkDir/rtl_scores_fp16.hex",
        '-do', 'run -all; quit -f'
    )
    $qkOutput = & $vsim @qkArgs 2>&1
    $qkOutput | Tee-Object -FilePath "$qkDir/modelsim.log"
    $qkText = $qkOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $qkText -notmatch 'SUPRA2_ATTN_QK_PASS queries=256 heads=9 keys=256' -or
        $qkText -match '\*\* Fatal|Errors: [1-9]') {
        throw "RTL attention QK regression failed; see $qkDir/modelsim.log"
    }

    & (Join-Path $PSScriptRoot 'run_attention_online_pv_batch.ps1') `
        -ModelSimRoot $ModelSimRoot `
        -QkvFile "$qprojDir/rtl_output_fp16.hex" `
        -ScoreFile "$qkDir/rtl_scores_fp16.hex" `
        -OutputDir "$pvDir"
    if ($LASTEXITCODE -ne 0) { throw 'Batched fused online softmax-to-PV regression failed' }

    python -m compiler.supra2.prepare_qlinear_projection `
        --layer "blocks.$Block.self_attn.proj" --fp16-hadamard-frontend `
        --raw-input-file "$pvDir/rtl_attention_fp16.hex" `
        --output-dir $projDir
    if ($LASTEXITCODE -ne 0) { throw 'Self-attention output projection fixture generation failed' }
    $selfProjArgs = @(
        '-c', '-work', $qprojLibrary, 'tb_supra2_qlinear_projection',
        '+M=256', '+N=576', '+K=576', '+H=64',
        "+RAW_INPUT_FILE=$projDir/raw_input_fp16.hex",
        "+EXPECTED_Q_FILE=$projDir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$projDir/rotated_fp16.hex",
        "+WEIGHT_FILE=$projDir/weight_int8.hex",
        "+ACC_FILE=$projDir/accumulator_int32.hex",
        "+OUTPUT_FILE=$projDir/output_fp16.hex",
        "+ACTUAL_OUTPUT_FILE=$projDir/rtl_output_fp16.hex",
        "+REFERENCE_OUTPUT_FILE=$projDir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$projDir/token_scale_fp32.hex",
        "+WEIGHT_SCALE_FILE=$projDir/weight_scale_fp32.hex",
        "+BIAS_FILE=$projDir/bias_fp32.hex",
        '-do', 'run -all; quit -f'
    )
    $selfProjOutput = & $vsim @selfProjArgs 2>&1
    $selfProjOutput | Tee-Object -FilePath "$projDir/modelsim.log"
    $selfProjText = $selfProjOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $selfProjText -notmatch 'SUPRA2_QPROJ_PASS M=256 N=576 K=576' -or
        $selfProjText -notmatch 'SUPRA2_QPROJ_FRONTEND_H_MISMATCHES 0' -or
        $selfProjText -notmatch 'SUPRA2_QPROJ_FRONTEND_Q_MISMATCHES 0' -or
        $selfProjText -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or
        $selfProjText -match '\*\* Fatal|Errors: [1-9]') {
        throw "RTL self-attention output projection failed; see $projDir/modelsim.log"
    }

    $residualFixtureArgs = @('--update-file', "$projDir/rtl_output_fp16.hex",
        '--gate-file', "$normDir/gate_sa_fp16.hex", '--output-dir', $residualDir,
        '--block', "$Block")
    if ($InputFile) { $residualFixtureArgs += @('--residual-file', $InputFile) }
    python -m compiler.supra2.prepare_self_attn_residual @residualFixtureArgs
    if ($LASTEXITCODE -ne 0) { throw 'Self-attention residual fixture generation failed' }
    if (-not (Test-Path $residualLibrary)) {
        & $vlib $residualLibrary
        if ($LASTEXITCODE -ne 0) { throw 'Residual vlib failed' }
    }
    & $vlog -sv -work $residualLibrary -f rtl/supra2/residual_gate.f
    if ($LASTEXITCODE -ne 0) { throw 'Residual RTL compile failed' }
    $residualArgs = @(
        '-c', '-work', $residualLibrary, 'tb_supra2_residual_gate',
        "+RESIDUAL_FILE=$residualDir/residual_fp16.hex",
        "+UPDATE_FILE=$residualDir/update_fp16.hex",
        "+GATE_FILE=$residualDir/gate_fp16.hex",
        "+EXPECTED_FILE=$residualDir/expected_fp16.hex",
        "+ACTUAL_FILE=$residualDir/rtl_actual_fp16.hex",
        '-do', 'run -all; quit -f'
    )
    $residualOutput = & $vsim @residualArgs 2>&1
    $residualOutput | Tee-Object -FilePath "$residualDir/modelsim.log"
    $residualText = $residualOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $residualText -notmatch 'SUPRA2_RESIDUAL_GATE_PASS rows=256 features=576' -or
        $residualText -match '\*\* Fatal|Errors: [1-9]') {
        throw "RTL self-attention residual failed; see $residualDir/modelsim.log"
    }
}
finally {
    Pop-Location
}
