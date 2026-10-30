param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [string]$QkvFile = 'build/supra2_qkv_after_layernorm/rtl_output_fp16.hex',
    [string]$ScoreFile = 'build/supra2_attention_qk/rtl_scores_fp16.hex'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vlib = Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog = Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim = Join-Path $ModelSimRoot 'win64\vsim.exe'
$license = Join-Path $ModelSimRoot 'LICENSE.TXT'
$library = 'build/modelsim/work_supra2_online_pv'
$outputDir = 'build/supra2_attention_online_pv'

foreach ($tool in @($vlib, $vlog, $vsim, $license)) {
    if (-not (Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license
    python -m compiler.supra2.prepare_attention_online_pv `
        --qkv-file $QkvFile --score-file $ScoreFile --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw 'Online PV fixture generation failed' }

    if (-not (Test-Path $library)) {
        & $vlib $library
        if ($LASTEXITCODE -ne 0) { throw 'Online PV vlib failed' }
    }
    & $vlog -sv -work $library -f rtl/supra2/attention_online_pv.f `
        *> "$outputDir/compile.log"
    if ($LASTEXITCODE -ne 0) {
        Get-Content "$outputDir/compile.log"
        throw 'Online PV RTL compile failed'
    }

    $simArgs = @(
        '-c', '-work', $library, 'tb_supra2_attention_online_pv',
        "+SCORE_FILE=$outputDir/scores_fp16.hex",
        "+VALUE_FILE=$outputDir/values_head_dim_key_fp16.hex",
        "+EXPECTED_FILE=$outputDir/expected_attention_fp16.hex",
        "+ACTUAL_FILE=$outputDir/rtl_attention_fp16.hex",
        '-do', 'run -all; quit -f'
    )
    $simOutput = & $vsim @simArgs 2>&1
    $simOutput | Tee-Object -FilePath "$outputDir/modelsim.log"
    $simText = $simOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $simText -notmatch 'SUPRA2_ATTN_ONLINE_PV_PASS queries=256 heads=9 keys=256' -or
        $simText -match '\*\* Fatal|Errors: [1-9]|Warnings: [1-9]') {
        throw "Online softmax-to-PV regression failed; see $outputDir/modelsim.log"
    }
}
finally {
    Pop-Location
}
