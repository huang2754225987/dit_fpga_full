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
$export = 'build/supra2_hw_export/export'
$tag = if ($Block -eq 0) { 'supra2' } else { "supra2_b$Block" }
$normDir = "build/${tag}_cross_q_norm"
$qDir = "build/${tag}_cross_q_proj"
$kvDir = "build/${tag}_cross_kv_proj"
$packedContextDir = "build/${tag}_cross_context_packed"
$qkDir = "build/${tag}_cross_attention_qk"
$pvDir = "build/${tag}_cross_attention_online_pv"
$projDir = "build/${tag}_cross_attention_proj"
$residualDir = "build/${tag}_cross_attention_residual"
$selfResidualDir = "build/${tag}_self_attn_residual"
$selfProjDir = "build/${tag}_self_attn_proj"
$adaDir = "build/${tag}_layernorm_modulate"
$mlpNormDir = "build/${tag}_block${Block}_mlp_norm"
$mlp0Dir = "build/${tag}_block${Block}_mlp0"
$geluDir = "build/${tag}_block${Block}_gelu"
$mlp2Dir = "build/${tag}_block${Block}_mlp2"
$mlpResidualDir = "build/${tag}_block${Block}_mlp_residual"
$qLib = 'build/modelsim/work_supra2_cross_qlinear'
$normLib = 'build/modelsim/work_supra2_cross_norm'
$qkLib = 'build/modelsim/work_supra2_cross_qk_tuned'
$pvLib = 'build/modelsim/work_supra2_cross_pv'

foreach ($tool in @($vlib, $vlog, $vsim, $license)) {
    if (-not (Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}
foreach ($file in @("$export/weights_quant_int8.npz", "$selfResidualDir/expected_fp16.hex",
                    'build/supra2_ctx_proj/projection/rtl_output_fp16.hex')) {
    if (-not (Test-Path $file)) { throw "Required preceding RTL artifact not found: $file" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license
    $adaFixtureArgs = @('--export-dir', $export, '--output-dir', $adaDir, '--block', "$Block")
    if ($InputFile) { $adaFixtureArgs += @('--input-file', $InputFile) }
    python -m compiler.supra2.prepare_layernorm_modulate @adaFixtureArgs
    if ($LASTEXITCODE -ne 0) { throw "Block-$Block AdaLN fixture generation failed" }
    python -m compiler.supra2.prepare_cross_attention_context --export-dir $export `
        --projection-file build/supra2_ctx_proj/projection/rtl_output_fp16.hex `
        --output-dir $packedContextDir
    if ($LASTEXITCODE -ne 0) { throw 'Masked cross-attention context packing failed' }
    $contextMeta = Get-Content "$packedContextDir/manifest.json" -Raw | ConvertFrom-Json
    $validKeys = [int]$contextMeta.valid_keys

    $selfResidualFixtureArgs = @('--update-file', "$selfProjDir/rtl_output_fp16.hex",
        '--gate-file', "$adaDir/gate_sa_fp16.hex", '--output-dir', $selfResidualDir,
        '--block', "$Block")
    if ($InputFile) { $selfResidualFixtureArgs += @('--residual-file', $InputFile) }
    python -m compiler.supra2.prepare_self_attn_residual @selfResidualFixtureArgs
    if ($LASTEXITCODE -ne 0) { throw 'Self-attention residual fixture failed' }
    $selfResidualLib = 'build/modelsim/work_supra2_cross_self_residual'
    if (-not (Test-Path $selfResidualLib)) { & $vlib $selfResidualLib; if ($LASTEXITCODE -ne 0) { throw 'Self residual vlib failed' } }
    & $vlog -sv -work $selfResidualLib -f rtl/supra2/residual_gate.f *> "$selfResidualDir/compile_cross.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$selfResidualDir/compile_cross.log"; throw 'Self residual compile failed' }
    $simArgs = @('-c','-work',$selfResidualLib,'tb_supra2_residual_gate',
        "+RESIDUAL_FILE=$selfResidualDir/residual_fp16.hex", "+UPDATE_FILE=$selfResidualDir/update_fp16.hex",
        "+GATE_FILE=$selfResidualDir/gate_fp16.hex", "+EXPECTED_FILE=$selfResidualDir/expected_fp16.hex",
        "+ACTUAL_FILE=$selfResidualDir/rtl_actual_fp16.hex", '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$selfResidualDir/modelsim_cross.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_RESIDUAL_GATE_PASS rows=256 features=576' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'Self-attention residual RTL failed'
    }

    python -m compiler.supra2.prepare_layernorm_only `
        --input-file "$selfResidualDir/rtl_actual_fp16.hex" `
        --output-dir $normDir --rows 256
    if ($LASTEXITCODE -ne 0) { throw 'Cross-attention query LayerNorm fixture failed' }
    if (-not (Test-Path $normLib)) { & $vlib $normLib; if ($LASTEXITCODE -ne 0) { throw 'LayerNorm vlib failed' } }
    & $vlog -sv -work $normLib -f rtl/supra2/layernorm_modulate.f *> "$normDir/compile.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$normDir/compile.log"; throw 'LayerNorm compile failed' }
    $simArgs = @('-c','-work',$normLib,'tb_supra2_layernorm_modulate',
        "+X_FILE=$normDir/x_fp16.hex", "+SHIFT_FILE=$normDir/shift_fp16.hex",
        "+SCALE_FILE=$normDir/scale_fp16.hex", "+EXPECTED_FILE=$normDir/reference_fp16.hex",
        "+QKV_FILE=$normDir/qkv_reference_fp16.hex", "+OUTPUT_FILE=$normDir/rtl_layernorm_fp16.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$normDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_LAYERNORM_MODULATE_PASS' -or $txt -match '\*\* Fatal|Errors: [1-9]') { throw 'Cross query LayerNorm failed' }

    foreach ($spec in @(
        @{ Layer="blocks.$Block.cross_attn.q"; Input="$normDir/rtl_layernorm_fp16.hex"; Dir=$qDir; N=576; M=256 },
        @{ Layer="blocks.$Block.cross_attn.kv"; Input="$packedContextDir/valid_context_fp16.hex"; Dir=$kvDir; N=1152; M=32 }
    )) {
        python -m compiler.supra2.prepare_qlinear_projection --export-dir $export `
            --layer $spec.Layer --fp16-hadamard-frontend --raw-input-file $spec.Input --output-dir $spec.Dir
        if ($LASTEXITCODE -ne 0) { throw "Fixture generation failed for $($spec.Layer)" }
        if (-not (Test-Path $qLib)) { & $vlib $qLib; if ($LASTEXITCODE -ne 0) { throw 'QLinear vlib failed' } }
        & $vlog -sv -work $qLib -f rtl/supra2/qlinear_projection.f *> "$($spec.Dir)/compile.log"
        if ($LASTEXITCODE -ne 0) { Get-Content "$($spec.Dir)/compile.log"; throw 'QLinear compile failed' }
        $simArgs = @('-c','-work',$qLib,'tb_supra2_qlinear_projection',
            "+M=$($spec.M)", "+N=$($spec.N)", '+K=576', '+H=64',
            "+RAW_INPUT_FILE=$($spec.Dir)/raw_input_fp16.hex",
            "+EXPECTED_Q_FILE=$($spec.Dir)/expected_quantized_int8.hex",
            "+EXPECTED_ROTATED_FILE=$($spec.Dir)/rotated_fp16.hex",
            "+WEIGHT_FILE=$($spec.Dir)/weight_int8.hex",
            "+ACC_FILE=$($spec.Dir)/accumulator_int32.hex",
            "+OUTPUT_FILE=$($spec.Dir)/output_fp16.hex",
            "+ACTUAL_OUTPUT_FILE=$($spec.Dir)/rtl_output_fp16.hex",
            "+REFERENCE_OUTPUT_FILE=$($spec.Dir)/reference_output_fp16.hex",
            "+TOKEN_SCALE_FILE=$($spec.Dir)/token_scale_fp32.hex",
            "+WEIGHT_SCALE_FILE=$($spec.Dir)/weight_scale_fp32.hex",
            "+BIAS_FILE=$($spec.Dir)/bias_fp32.hex", '-do','run -all; quit -f')
        $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$($spec.Dir)/modelsim.log"
        $txt = $out -join "`n"
        if ($LASTEXITCODE -ne 0 -or $txt -notmatch "SUPRA2_QPROJ_PASS M=$($spec.M) N=$($spec.N) K=576" -or
            $txt -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
            throw "RTL cross-attention projection failed: $($spec.Layer)"
        }
    }

    python -m compiler.supra2.prepare_cross_attention_qk --export-dir $export `
        --q-file "$qDir/rtl_output_fp16.hex" --kv-file "$kvDir/rtl_output_fp16.hex" --output-dir $qkDir
    if ($LASTEXITCODE -ne 0) { throw 'Cross QK fixture generation failed' }
    if (-not (Test-Path $qkLib)) { & $vlib $qkLib; if ($LASTEXITCODE -ne 0) { throw 'QK vlib failed' } }
    & $vlog -sv -work $qkLib -f rtl/supra2/attention_qk.f *> "$qkDir/compile.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$qkDir/compile.log"; throw 'QK compile failed' }
    $simArgs = @('-c','-work',$qkLib,'tb_supra2_attention_qk','-gKEYS=32',
        "+Q_FILE=$qkDir/q_fp16.hex", "+K_FILE=$qkDir/k_fp16.hex",
        "+EXPECTED_FILE=$qkDir/expected_scores_fp16.hex", "+ACTUAL_FILE=$qkDir/rtl_scores_fp16.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$qkDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_ATTN_QK_PASS queries=256 heads=9 keys=32' -or $txt -match '\*\* Fatal|Errors: [1-9]') { throw 'Cross-attention QK RTL failed' }

    python -m compiler.supra2.prepare_cross_attention_online_pv --export-dir $export `
        --kv-file "$kvDir/rtl_output_fp16.hex" --score-file "$qkDir/rtl_scores_fp16.hex" --output-dir $pvDir
    if ($LASTEXITCODE -ne 0) { throw 'Cross online PV fixture generation failed' }
    if (-not (Test-Path $pvLib)) { & $vlib $pvLib; if ($LASTEXITCODE -ne 0) { throw 'PV vlib failed' } }
    & $vlog -sv -work $pvLib -f rtl/supra2/attention_online_pv_batch.f *> "$pvDir/compile.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$pvDir/compile.log"; throw 'PV compile failed' }
    $simArgs = @('-c','-work',$pvLib,'tb_supra2_attention_online_pv_batch',
        '-gKEYS=32', "-gVALID_KEYS=$validKeys",
        "+SCORE_FILE=$pvDir/scores_fp16.hex", "+VALUE_FILE=$pvDir/values_head_dim_key_fp16.hex",
        "+EXPECTED_FILE=$pvDir/expected_attention_fp16.hex", "+ACTUAL_FILE=$pvDir/rtl_attention_fp16.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$pvDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_ATTN_ONLINE_PV_BATCH_PASS queries=256 heads=9 keys=32' -or $txt -match '\*\* Fatal|Errors: [1-9]') { throw 'Cross-attention online PV RTL failed' }

    python -m compiler.supra2.prepare_qlinear_projection --export-dir $export `
        --layer "blocks.$Block.cross_attn.proj" --fp16-hadamard-frontend `
        --raw-input-file "$pvDir/rtl_attention_fp16.hex" --output-dir $projDir
    if ($LASTEXITCODE -ne 0) { throw 'Cross-attention output projection fixture failed' }
    $simArgs = @('-c','-work',$qLib,'tb_supra2_qlinear_projection',
        '+M=256','+N=576','+K=576','+H=64',
        "+RAW_INPUT_FILE=$projDir/raw_input_fp16.hex",
        "+EXPECTED_Q_FILE=$projDir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$projDir/rotated_fp16.hex",
        "+WEIGHT_FILE=$projDir/weight_int8.hex", "+ACC_FILE=$projDir/accumulator_int32.hex",
        "+OUTPUT_FILE=$projDir/output_fp16.hex", "+ACTUAL_OUTPUT_FILE=$projDir/rtl_output_fp16.hex",
        "+REFERENCE_OUTPUT_FILE=$projDir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$projDir/token_scale_fp32.hex",
        "+WEIGHT_SCALE_FILE=$projDir/weight_scale_fp32.hex", "+BIAS_FILE=$projDir/bias_fp32.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$projDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_QPROJ_PASS M=256 N=576 K=576' -or
        $txt -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'Cross-attention output projection RTL failed'
    }

    python -m compiler.supra2.prepare_cross_attention_residual `
        --residual-file "$selfResidualDir/rtl_actual_fp16.hex" `
        --update-file "$projDir/rtl_output_fp16.hex" --output-dir $residualDir
    if ($LASTEXITCODE -ne 0) { throw 'Cross-attention residual fixture failed' }
    $residualLib = 'build/modelsim/work_supra2_cross_residual'
    if (-not (Test-Path $residualLib)) { & $vlib $residualLib; if ($LASTEXITCODE -ne 0) { throw 'Residual vlib failed' } }
    & $vlog -sv -work $residualLib -f rtl/supra2/residual_gate.f *> "$residualDir/compile.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$residualDir/compile.log"; throw 'Residual compile failed' }
    $simArgs = @('-c','-work',$residualLib,'tb_supra2_residual_gate',
        "+RESIDUAL_FILE=$residualDir/residual_fp16.hex", "+UPDATE_FILE=$residualDir/update_fp16.hex",
        "+GATE_FILE=$residualDir/gate_fp16.hex", "+EXPECTED_FILE=$residualDir/expected_fp16.hex",
        "+ACTUAL_FILE=$residualDir/rtl_actual_fp16.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$residualDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_RESIDUAL_GATE_PASS rows=256 features=576' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'Cross-attention residual RTL failed'
    }

    python -m compiler.supra2.prepare_layernorm_only --input-file "$residualDir/rtl_actual_fp16.hex" `
        --output-dir $mlpNormDir --rows 256 --shift-file "$adaDir/shift_mlp_fp16.hex" `
        --scale-file "$adaDir/scale_mlp_fp16.hex"
    if ($LASTEXITCODE -ne 0) { throw 'MLP LayerNorm/AdaLN fixture failed' }
    $simArgs = @('-c','-work',$normLib,'tb_supra2_layernorm_modulate',
        "+X_FILE=$mlpNormDir/x_fp16.hex", "+SHIFT_FILE=$mlpNormDir/shift_fp16.hex",
        "+SCALE_FILE=$mlpNormDir/scale_fp16.hex", "+EXPECTED_FILE=$mlpNormDir/reference_fp16.hex",
        "+QKV_FILE=$mlpNormDir/qkv_reference_fp16.hex", "+OUTPUT_FILE=$mlpNormDir/rtl_layernorm_fp16.hex",
        '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$mlpNormDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_LAYERNORM_MODULATE_PASS' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'MLP LayerNorm/AdaLN RTL failed'
    }

    python -m compiler.supra2.prepare_qlinear_projection --export-dir $export `
        --layer "blocks.$Block.mlp.0" --fp16-hadamard-frontend `
        --raw-input-file "$mlpNormDir/rtl_layernorm_fp16.hex" --output-dir $mlp0Dir
    if ($LASTEXITCODE -ne 0) { throw 'MLP.0 fixture generation failed' }
    $simArgs = @('-c','-work',$qLib,'tb_supra2_qlinear_projection',
        '+M=256','+N=2304','+K=576','+H=64',
        "+RAW_INPUT_FILE=$mlp0Dir/raw_input_fp16.hex", "+EXPECTED_Q_FILE=$mlp0Dir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$mlp0Dir/rotated_fp16.hex", "+WEIGHT_FILE=$mlp0Dir/weight_int8.hex",
        "+ACC_FILE=$mlp0Dir/accumulator_int32.hex", "+OUTPUT_FILE=$mlp0Dir/output_fp16.hex",
        "+ACTUAL_OUTPUT_FILE=$mlp0Dir/rtl_output_fp16.hex", "+REFERENCE_OUTPUT_FILE=$mlp0Dir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$mlp0Dir/token_scale_fp32.hex", "+WEIGHT_SCALE_FILE=$mlp0Dir/weight_scale_fp32.hex",
        "+BIAS_FILE=$mlp0Dir/bias_fp32.hex", '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$mlp0Dir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_QPROJ_PASS M=256 N=2304 K=576' -or
        $txt -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'MLP.0 RTL projection failed'
    }

    python -m compiler.supra2.prepare_gelu_tanh --input-file "$mlp0Dir/rtl_output_fp16.hex" `
        --output-dir $geluDir
    if ($LASTEXITCODE -ne 0) { throw 'GELU fixture/LUT generation failed' }
    $geluLib = 'build/modelsim/work_supra2_gelu_tanh'
    if (-not (Test-Path $geluLib)) { & $vlib $geluLib; if ($LASTEXITCODE -ne 0) { throw 'GELU vlib failed' } }
    & $vlog -sv -work $geluLib -f rtl/supra2/gelu_tanh.f *> "$geluDir/compile.log"
    if ($LASTEXITCODE -ne 0) { Get-Content "$geluDir/compile.log"; throw 'GELU compile failed' }
    $simArgs = @('-c','-work',$geluLib,'tb_supra2_gelu_tanh',
        "+INPUT_FILE=$mlp0Dir/rtl_output_fp16.hex", "+EXPECTED_FILE=$geluDir/expected_fp16.hex",
        "+ACTUAL_FILE=$geluDir/rtl_gelu_fp16.hex", '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$geluDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_GELU_TANH_PASS' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'GELU(tanh) RTL failed'
    }

    python -m compiler.supra2.prepare_qlinear_projection --export-dir $export `
        --layer "blocks.$Block.mlp.2" --fp16-hadamard-frontend `
        --raw-input-file "$geluDir/rtl_gelu_fp16.hex" --output-dir $mlp2Dir
    if ($LASTEXITCODE -ne 0) { throw 'MLP.2 fixture generation failed' }
    $simArgs = @('-c','-work',$qLib,'tb_supra2_qlinear_projection',
        '+M=256','+N=576','+K=2304','+H=256',
        "+RAW_INPUT_FILE=$mlp2Dir/raw_input_fp16.hex", "+EXPECTED_Q_FILE=$mlp2Dir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$mlp2Dir/rotated_fp16.hex", "+WEIGHT_FILE=$mlp2Dir/weight_int8.hex",
        "+ACC_FILE=$mlp2Dir/accumulator_int32.hex", "+OUTPUT_FILE=$mlp2Dir/output_fp16.hex",
        "+ACTUAL_OUTPUT_FILE=$mlp2Dir/rtl_output_fp16.hex", "+REFERENCE_OUTPUT_FILE=$mlp2Dir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$mlp2Dir/token_scale_fp32.hex", "+WEIGHT_SCALE_FILE=$mlp2Dir/weight_scale_fp32.hex",
        "+BIAS_FILE=$mlp2Dir/bias_fp32.hex", '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$mlp2Dir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_QPROJ_PASS M=256 N=576 K=2304' -or
        $txt -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'MLP.2 RTL projection failed'
    }

    python -m compiler.supra2.prepare_self_attn_residual --export-dir $export `
        --residual-file "$residualDir/rtl_actual_fp16.hex" `
        --update-file "$mlp2Dir/rtl_output_fp16.hex" --gate-file "$adaDir/gate_mlp_fp16.hex" `
        --output-dir $mlpResidualDir --block $Block
    if ($LASTEXITCODE -ne 0) { throw 'MLP gated residual fixture failed' }
    $simArgs = @('-c','-work',$residualLib,'tb_supra2_residual_gate',
        "+RESIDUAL_FILE=$mlpResidualDir/residual_fp16.hex", "+UPDATE_FILE=$mlpResidualDir/update_fp16.hex",
        "+GATE_FILE=$mlpResidualDir/gate_fp16.hex", "+EXPECTED_FILE=$mlpResidualDir/expected_fp16.hex",
        "+ACTUAL_FILE=$mlpResidualDir/rtl_block${Block}_output_fp16.hex", '-do','run -all; quit -f')
    $out = & $vsim @simArgs 2>&1; $out | Tee-Object -FilePath "$mlpResidualDir/modelsim.log"
    $txt = $out -join "`n"
    if ($LASTEXITCODE -ne 0 -or $txt -notmatch 'SUPRA2_RESIDUAL_GATE_PASS rows=256 features=576' -or $txt -match '\*\* Fatal|Errors: [1-9]') {
        throw 'MLP gated residual RTL failed'
    }
    python -m compiler.supra2.verify_block0_output --actual-file "$mlpResidualDir/rtl_block${Block}_output_fp16.hex" --export-dir $export --block $Block --allow-missing
    if ($LASTEXITCODE -ne 0) { throw "RTL block-$Block output did not meet archive golden gate" }
}
finally { Pop-Location }
