param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [ValidateSet('all', 'x_embed', 'qkv', 'mlp2')]
    [string]$Only = 'all',
    [string]$LicenseFile = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vlib = Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog = Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim = Join-Path $ModelSimRoot 'win64\vsim.exe'
$license = if ($LicenseFile) { $LicenseFile } else { Join-Path $ModelSimRoot 'LICENSE.TXT' }
$workLibrary = 'build/modelsim/work_supra2_frontend'

foreach ($tool in @($vlib, $vlog, $vsim, $license)) {
    if (-not (Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license

    if (-not (Test-Path $workLibrary)) {
        & $vlib $workLibrary
        if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    }

    & $vlog -sv -work $workLibrary -f rtl/supra2/qlinear_projection.f
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }

    $cases = @(
        @{ Name = 'x_embed'; Layer = 'x_embed'; N = 576; K = 16; H = 16;
           OutputDir = 'build/supra2_x_embed_qlinear'; Fixture = 'prepare_x_embed_qlinear_projection' },
        @{ Name = 'qkv'; Layer = 'blocks.0.self_attn.qkv'; N = 1728; K = 576; H = 64;
           OutputDir = 'build/supra2_qkv_projection_fp32_frontend'; Fixture = 'prepare_qlinear_projection' },
        @{ Name = 'mlp2'; Layer = 'blocks.0.mlp.2'; N = 576; K = 2304; H = 256;
           OutputDir = 'build/supra2_mlp2_projection_fp32_frontend'; Fixture = 'prepare_qlinear_projection' }
    )
    if ($Only -ne 'all') {
        $cases = @($cases | Where-Object { $_.Name -eq $Only })
    }

    foreach ($case in $cases) {
        $outputDir = $case.OutputDir
        if ($case.Fixture -eq 'prepare_x_embed_qlinear_projection') {
            python -m compiler.supra2.prepare_x_embed_qlinear_projection --output-dir $outputDir
        }
        else {
            python -m compiler.supra2.prepare_qlinear_projection `
                --layer $case.Layer --fp16-hadamard-frontend --output-dir $outputDir
        }
        if ($LASTEXITCODE -ne 0) { throw "Fixture generation failed for $($case.Name)" }

        $simArgs = @(
            '-c', '-work', $workLibrary, 'tb_supra2_qlinear_projection',
            '+M=256', "+N=$($case.N)", "+K=$($case.K)", "+H=$($case.H)",
            "+RAW_INPUT_FILE=$outputDir/raw_input_fp16.hex",
            "+EXPECTED_Q_FILE=$outputDir/expected_quantized_int8.hex",
            "+EXPECTED_ROTATED_FILE=$outputDir/rotated_fp16.hex",
            "+WEIGHT_FILE=$outputDir/weight_int8.hex",
            "+ACC_FILE=$outputDir/accumulator_int32.hex",
            "+OUTPUT_FILE=$outputDir/output_fp16.hex",
            "+ACTUAL_OUTPUT_FILE=$outputDir/rtl_output_fp16.hex",
            "+REFERENCE_OUTPUT_FILE=$outputDir/reference_output_fp16.hex",
            "+TOKEN_SCALE_FILE=$outputDir/token_scale_fp32.hex",
            "+WEIGHT_SCALE_FILE=$outputDir/weight_scale_fp32.hex",
            "+BIAS_FILE=$outputDir/bias_fp32.hex",
            '-do', 'run -all; quit -f'
        )
        $simOutput = & $vsim @simArgs 2>&1
        $simOutput | Tee-Object -FilePath "$outputDir/modelsim.log"
        $simText = $simOutput -join "`n"
        if ($LASTEXITCODE -ne 0 -or
            $simText -notmatch "SUPRA2_QPROJ_PASS M=256 N=$($case.N) K=$($case.K)" -or
            $simText -notmatch 'SUPRA2_QPROJ_FRONTEND_H_MISMATCHES 0' -or
            $simText -notmatch 'SUPRA2_QPROJ_FRONTEND_Q_MISMATCHES 0' -or
            $simText -notmatch 'SUPRA2_QPROJ_SOFTWARE_CHAIN_MISMATCHES 0' -or
            $simText -match '\*\* Fatal|Errors: [1-9]') {
            throw "RTL regression failed for $($case.Name); see $outputDir/modelsim.log"
        }
    }
}
finally {
    Pop-Location
}
