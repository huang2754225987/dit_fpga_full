param(
    [string]$QuestaRoot = 'D:\soft\questa\win64',
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [string]$InputFile = 'build\supra2_b13_block13_mlp_residual\rtl_block13_output_fp16.hex'
)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
$env:LM_LICENSE_FILE = $license
$env:MGLS_LICENSE_FILE = $license
$env:PATH = "$QuestaRoot;$env:PATH"
Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot 'run_final_head_regression.ps1') -ModelSimRoot $ModelSimRoot -InputFile $InputFile
    if ($LASTEXITCODE -ne 0) { throw 'The direct-array baseline failed' }
    $baselineLog = Get-Content build/supra2_final_head/modelsim.log -Raw
    if ($baselineLog -notmatch 'SUPRA2_FINAL_HEAD_PASS' -or
        -not (Test-Path -LiteralPath build/supra2_final_head/rtl_tokens_fp16.hex)) {
        throw 'The direct-array baseline did not pass or produce its token fixture'
    }
    $work = 'build/questa/work_supra2_final_head_projection'
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    & (Join-Path $QuestaRoot 'vlib.exe') $work
    if ($LASTEXITCODE -ne 0) { throw 'vlib failed' }
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work $work -f rtl/supra2/final_head_projection.f
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed' }
    $sim = & (Join-Path $QuestaRoot 'vsim.exe') -c -work $work tb_supra2_final_head_projection `
        -do 'run -all; quit -f' 2>&1
    $sim | Tee-Object -FilePath build/supra2_final_head/projection_questa.log
    $text = $sim -join "`n"
    if ($LASTEXITCODE -ne 0 -or $text -notmatch 'SUPRA2_FINAL_HEAD_PROJECTION_PASS' -or
        $text -match '\*\* Fatal|Errors: [1-9]') { throw 'Projection Questa regression failed' }
}
finally { Pop-Location }
