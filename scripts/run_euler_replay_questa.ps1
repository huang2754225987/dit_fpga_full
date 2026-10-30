$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $repo "build\supra2_euler_replay\questa_runs\$runTag"
$fixtureDir = Join-Path $runDir 'fixtures'
foreach ($tool in @('vlib.exe', 'vlog.exe', 'vsim.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $questa $tool))) {
        throw "Questa executable not found: $(Join-Path $questa $tool)"
    }
}
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
$env:LM_LICENSE_FILE = $license
$env:MGLS_LICENSE_FILE = $license
$env:PATH = "$questa;$env:PATH"
New-Item -ItemType Directory -Path $runDir -Force | Out-Null
Push-Location $runDir
try {
    python (Join-Path $repo 'compiler\supra2\prepare_euler_replay.py') `
        --output-dir $fixtureDir
    if ($LASTEXITCODE -ne 0) { throw 'Euler fixture generation failed' }
    & (Join-Path $questa 'vlib.exe') work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    $sources = @(
        (Join-Path $repo 'rtl\pkg\fp16_utils_pkg.sv'),
        (Join-Path $repo 'rtl\supra2\euler_update.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_euler_replay.sv')
    )
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $runDir 'euler_replay_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_euler_replay `
        "+Z0_FILE=$fixtureDir/z0_fp16.hex" `
        "+VELOCITY_FILE=$fixtureDir/velocity_steps_fp16.hex" `
        "+EXPECTED_FILE=$fixtureDir/z_final_fp16.hex" `
        "+REFERENCE_FILE=$fixtureDir/reference_final_fp16.hex" `
        -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    $logText = Get-Content -LiteralPath $logPath -Raw
    if ($logText -notmatch 'SUPRA2_EULER_REPLAY_PASS steps=50 elements=4096' -or
        $logText -notmatch 'SUPRA2_EULER_REPLAY_SOFTWARE_MISMATCHES 0' -or
        $logText -match '\*\* Fatal|Errors: [1-9]') {
        throw "Questa completed without the required Euler replay PASS (see $logPath)"
    }
    "Questa 50-step golden-velocity Euler replay PASS; log: $logPath"
} finally {
    Pop-Location
}
