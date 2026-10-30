param(
    [string]$QuestaRoot='D:\soft\questa\win64',
    [string]$InputFile='build\supra2_b13_block13_mlp_residual\rtl_block13_output_fp16.hex'
)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$license='C:\ProgramData\MentorGraphics\License\license.dat'
if(-not(Test-Path -LiteralPath $license)){throw "Questa license not found: $license"}
$env:LM_LICENSE_FILE=$license; $env:MGLS_LICENSE_FILE=$license; $env:PATH="$QuestaRoot;$env:PATH"
python -m compiler.supra2.prepare_final_head --input-file $InputFile
if($LASTEXITCODE -ne 0){throw 'Final-head fixture generation failed'}
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir=Join-Path $repo "build\supra2_bridge_final_head\questa_runs\$runTag"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$fixtureDir=Join-Path $workDir 'build\supra2_final_head'
New-Item -ItemType Directory -Force -Path $fixtureDir | Out-Null
Copy-Item -Path (Join-Path $repo 'build\supra2_final_head\*') -Destination $fixtureDir -Recurse -Force
$lutDir=Join-Path $workDir 'rtl\supra2'; New-Item -ItemType Directory -Force -Path $lutDir | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'rtl\supra2\gelu_tanh_lut_fp16.hex') -Destination $lutDir
Push-Location $workDir
try {
    & (Join-Path $QuestaRoot 'vlib.exe') work
    if($LASTEXITCODE -ne 0){throw 'vlib failed'}
    $sources=@(Get-Content (Join-Path $repo 'rtl\supra2\supra2_dit_compute_bridge.f') |
        Where-Object {$_ -and -not $_.StartsWith('#')} |
        ForEach-Object {Join-Path $repo ($_ -replace '/', '\')})
    $sources+=Join-Path $repo 'rtl\sim\tb_supra2_dit_compute_bridge_final_head.sv'
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work work @sources
    if($LASTEXITCODE -ne 0){throw 'vlog failed'}
    $log=Join-Path $workDir 'bridge_final_head_questa.log'
    $sim=& (Join-Path $QuestaRoot 'vsim.exe') -suppress 7061 -c -lib work `
        tb_supra2_dit_compute_bridge_final_head -do 'run -all; quit -f' 2>&1
    $exit=$LASTEXITCODE; $sim | Set-Content -LiteralPath $log; $sim
    $text=$sim -join "`n"
    if($exit -ne 0 -or $text -notmatch 'SUPRA2_BRIDGE_FINAL_HEAD_CFG_PASS' -or
       $text -match '(?m)^# \*\* Fatal' -or
       $text -match '(?m)^# Errors: [1-9]'){throw "Integrated final-head/CFG Questa test failed (see $log)"}
    "Integrated final-head/CFG Questa PASS; log: $log"
} finally { Pop-Location }
