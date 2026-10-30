param(
    [string]$QuestaRoot = 'D:\soft\questa\win64',
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
    python -m compiler.supra2.prepare_final_head --input-file $InputFile
    if ($LASTEXITCODE -ne 0) { throw 'Final-head fixture generation failed' }
    $work = 'build/questa/work_supra2_final_head_datapath'
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    & (Join-Path $QuestaRoot 'vlib.exe') $work
    if ($LASTEXITCODE -ne 0) { throw 'vlib failed' }
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work $work -f rtl/supra2/final_head_datapath.f
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed' }
    $sim = & (Join-Path $QuestaRoot 'vsim.exe') -c -work $work tb_supra2_final_head_datapath `
        -do 'run -all; quit -f' 2>&1
    $sim | Tee-Object -FilePath build/supra2_final_head/datapath_questa.log
    $text = $sim -join "`n"
    if ($LASTEXITCODE -ne 0 -or $text -notmatch 'SUPRA2_FINAL_HEAD_DATAPATH_PASS' -or
        $text -match '\*\* Fatal|Errors: [1-9]') { throw 'Final-head datapath Questa regression failed' }
}
finally { Pop-Location }
