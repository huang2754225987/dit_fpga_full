$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa='D:\soft\questa\win64'
$env:LM_LICENSE_FILE='C:\ProgramData\MentorGraphics\License\license.dat'
$env:MGLS_LICENSE_FILE=$env:LM_LICENSE_FILE
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir=Join-Path $repo "build\fp32_shift_right_jam\questa_runs\$runTag"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if($LASTEXITCODE -ne 0){throw 'vlib failed'}
    & (Join-Path $questa 'vlog.exe') -sv -work work `
        (Join-Path $repo 'rtl/pkg/fp16_utils_pkg.sv') `
        (Join-Path $repo 'rtl/sim/tb_fp32_shift_right_jam.sv')
    if($LASTEXITCODE -ne 0){throw 'vlog failed'}
    $sim=& (Join-Path $questa 'vsim.exe') -c -lib work tb_fp32_shift_right_jam -do 'run -all; quit -f' 2>&1
    $simExit=$LASTEXITCODE
    $sim | Set-Content 'shift_right_jam_questa.log'
    $sim
    $simText=$sim -join "`n"
    if($simExit -ne 0 -or $simText -notmatch 'FP32_SHIFT_RIGHT_JAM_PASS cases=24576' -or
        $simText -match '\*\* (Fatal|Error)|Errors: [1-9]'){throw 'FP32 jam regression failed'}
    "Questa FP32 jam regression PASS; logs: $workDir"
}finally {Pop-Location}
