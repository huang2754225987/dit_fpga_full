$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa='D:\soft\questa\win64'
$env:LM_LICENSE_FILE='C:\ProgramData\MentorGraphics\License\license.dat'
$env:MGLS_LICENSE_FILE=$env:LM_LICENSE_FILE
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir=Join-Path $repo "build\fp32_add_rne_pipeline\questa_runs\$runTag"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if($LASTEXITCODE -ne 0){throw 'vlib failed'}
    & (Join-Path $questa 'vlog.exe') -sv -work work `
        (Join-Path $repo 'rtl\pkg\fp16_utils_pkg.sv') `
        (Join-Path $repo 'rtl\supra2\fp32_add_rne_pipeline.sv') `
        (Join-Path $repo 'rtl\sim\tb_fp32_add_rne_pipeline.sv')
    if($LASTEXITCODE -ne 0){throw 'vlog failed'}
    $sim=& (Join-Path $questa 'vsim.exe') -c -lib work tb_fp32_add_rne_pipeline `
        -do 'run -all; quit -f' 2>&1
    $simExit=$LASTEXITCODE
    $sim | Tee-Object -FilePath 'fp32_add_rne_pipeline_questa.log'
    $text=$sim -join "`n"
    if($simExit -ne 0 -or $text -notmatch 'FP32_ADD_RNE_PIPELINE_PASS cases=20000 latency=8' -or
        $text -match '\*\* (Fatal|Error)|Errors: [1-9]') { throw 'FP32 pipeline Questa regression failed' }
    "FP32 adder pipeline Questa PASS; log: $workDir\fp32_add_rne_pipeline_questa.log"
} finally { Pop-Location }
