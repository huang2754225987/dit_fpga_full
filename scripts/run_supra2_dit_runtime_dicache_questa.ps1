$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_dit_runtime_dicache\questa_runs\$runTag"
$layout = Join-Path $repo 'build\supra2_dit_ddr_image\supra2_dit_ddr_layout.sv'
foreach ($tool in @('vlib.exe', 'vlog.exe', 'vsim.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $questa $tool))) {
        throw "Questa executable not found: $(Join-Path $questa $tool)"
    }
}
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
if (-not (Test-Path -LiteralPath $layout)) { throw "DDR layout not found: $layout" }
$env:LM_LICENSE_FILE = $license
$env:MGLS_LICENSE_FILE = $license
$env:PATH = "$questa;$env:PATH"
New-Item -ItemType Directory -Path $workDir -Force | Out-Null
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    $sources = @(
        $layout,
        (Join-Path $repo 'rtl\pkg\fp16_utils_pkg.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dicache_online_policy.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dicache_dcta_32lane.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dicache_residual_cache.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dit_runtime_controller.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_dit_runtime_dicache.sv')
    )
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $workDir 'runtime_dicache_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_dit_runtime_dicache -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
            -Pattern 'SUPRA2_DIT_RUNTIME_DICACHE_TOGGLE_PASS enabled_blocks=750 disabled_blocks=828 disabled_probes=0' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    "Questa runtime DiCache scheduling simulation PASS; log: $logPath"
} finally {
    Pop-Location
}
