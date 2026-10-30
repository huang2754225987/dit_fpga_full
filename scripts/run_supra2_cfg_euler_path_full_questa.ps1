$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_cfg_euler_path_full\questa_runs\$runTag"
foreach ($tool in @('vlib.exe', 'vlog.exe', 'vsim.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $questa $tool))) {
        throw "Questa executable not found: $(Join-Path $questa $tool)"
    }
}
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
$env:LM_LICENSE_FILE = $license
$env:MGLS_LICENSE_FILE = $license
$env:PATH = "$questa;$env:PATH"
New-Item -ItemType Directory -Path $workDir -Force | Out-Null
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    $sources = @(
        (Join-Path $repo 'rtl\pkg\fp16_utils_pkg.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_cfg_combine_32lane.sv'),
        (Join-Path $repo 'rtl\supra2\euler_update.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_cfg_euler_path.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_cfg_euler_path_full.sv')
    )
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $workDir 'cfg_euler_path_full_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_cfg_euler_path_full -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
            -Pattern 'SUPRA2_CFG_EULER_PATH_FULL_PASS elements=4096 beats=128 guidance=3' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    "Questa full-size CFG-to-Euler simulation PASS; log: $logPath"
} finally {
    Pop-Location
}
