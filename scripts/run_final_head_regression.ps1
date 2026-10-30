param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [string]$InputFile = 'build\supra2_b13_block13_mlp_residual\rtl_block13_output_fp16.hex'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$license = Join-Path $ModelSimRoot 'LICENSE.TXT'
Push-Location $repoRoot
try {
    if (-not (Test-Path $license)) { throw "Required ModelSim license not found: $license" }
    python -m compiler.supra2.prepare_final_head --input-file $InputFile
    if (-not $?) { throw 'Failed to prepare the final-head fixture' }

    $work = 'build/modelsim/work_supra2_final_head'
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $env:LM_LICENSE_FILE
    $env:PATH = (Join-Path $ModelSimRoot 'win64') + ';' + $env:PATH

    & (Join-Path $ModelSimRoot 'win64\vlib.exe') $work
    if ($LASTEXITCODE -ne 0) { throw 'vlib failed' }
    & (Join-Path $ModelSimRoot 'win64\vlog.exe') -sv -work $work -f rtl/supra2/final_head.f
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed' }
    $simOutput = & (Join-Path $ModelSimRoot 'win64\vsim.exe') -c -work $work tb_supra2_final_head `
        '+ACTIVATION_FILE=build/supra2_final_head/normalized_modulated_fp16.hex' `
        '+WEIGHT_FILE=build/supra2_final_head/linear_weight_fp16.hex' `
        '+BIAS_FILE=build/supra2_final_head/linear_bias_fp16.hex' `
        '+EXPECTED_FILE=build/supra2_final_head/reference_tokens_fp16.hex' `
        '+ACTUAL_FILE=build/supra2_final_head/rtl_tokens_fp16.hex' `
        -do 'run -all; quit -f' 2>&1
    $simOutput | Tee-Object -FilePath build/supra2_final_head/modelsim.log
    $simText = $simOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or $simText -notmatch 'SUPRA2_FINAL_HEAD_PASS' -or
        $simText -match '\*\* Fatal|Errors: [1-9]') {
        throw 'Final-head ModelSim regression failed'
    }
}
finally {
    Pop-Location
}
