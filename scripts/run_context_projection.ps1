param([string]$ModelSimRoot='D:\soft\modelsim')

$ErrorActionPreference='Stop'
$repoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vlib=Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog=Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim=Join-Path $ModelSimRoot 'win64\vsim.exe'
$license=Join-Path $ModelSimRoot 'LICENSE.TXT'
$library='build/modelsim/work_supra2_ctx_proj'
$outputDir='build/supra2_ctx_proj'

foreach($tool in @($vlib,$vlog,$vsim,$license)) {
    if(-not(Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE=$license
    $env:MGLS_LICENSE_FILE=$license
    python -m compiler.supra2.prepare_context_projection --output-dir $outputDir
    if($LASTEXITCODE -ne 0) { throw 'Context projection input fixture generation failed' }
    python -m compiler.supra2.prepare_qlinear_projection `
        --layer ctx_proj --fp16-hadamard-frontend `
        --raw-input-file "$outputDir/raw_input_fp16.hex" `
        --output-dir "$outputDir/projection"
    if($LASTEXITCODE -ne 0) { throw 'ctx_proj QLinear fixture generation failed' }

    if(-not(Test-Path $library)) {
        & $vlib $library
        if($LASTEXITCODE -ne 0) { throw 'ctx_proj vlib failed' }
    }
    & $vlog -sv -work $library -f rtl/supra2/qlinear_projection.f `
        *> "$outputDir/compile.log"
    if($LASTEXITCODE -ne 0) {
        Get-Content "$outputDir/compile.log"
        throw 'ctx_proj RTL compile failed'
    }
    $projectionDir="$outputDir/projection"
    $simArgs=@(
        '-c','-work',$library,'tb_supra2_qlinear_projection',
        '+M=128','+N=576','+K=768','+H=256',
        "+RAW_INPUT_FILE=$projectionDir/raw_input_fp16.hex",
        "+EXPECTED_Q_FILE=$projectionDir/expected_quantized_int8.hex",
        "+EXPECTED_ROTATED_FILE=$projectionDir/rotated_fp16.hex",
        "+WEIGHT_FILE=$projectionDir/weight_int8.hex",
        "+ACC_FILE=$projectionDir/accumulator_int32.hex",
        "+OUTPUT_FILE=$projectionDir/output_fp16.hex",
        "+ACTUAL_OUTPUT_FILE=$projectionDir/rtl_output_fp16.hex",
        "+REFERENCE_OUTPUT_FILE=$projectionDir/reference_output_fp16.hex",
        "+TOKEN_SCALE_FILE=$projectionDir/token_scale_fp32.hex",
        "+WEIGHT_SCALE_FILE=$projectionDir/weight_scale_fp32.hex",
        "+BIAS_FILE=$projectionDir/bias_fp32.hex",
        '-do','run -all; quit -f'
    )
    $simOutput=& $vsim @simArgs 2>&1
    $simOutput | Tee-Object -FilePath "$outputDir/modelsim.log"
    $simText=$simOutput -join "`n"
    if($LASTEXITCODE -ne 0 -or
       $simText -notmatch 'SUPRA2_QPROJ_PASS M=128 N=576 K=768' -or
       $simText -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or
       $simText -match '\*\* Fatal|Errors: [1-9]|Warnings: [1-9]') {
        throw "ctx_proj RTL regression failed; see $outputDir/modelsim.log"
    }
    python -m compiler.supra2.verify_context_projection `
        --reference-file "$outputDir/expected_block_context_fp16.hex" `
        --actual-file "$projectionDir/rtl_output_fp16.hex"
    if($LASTEXITCODE -ne 0) { throw 'RTL ctx_proj disagrees with block-0 context golden' }
}
finally { Pop-Location }
