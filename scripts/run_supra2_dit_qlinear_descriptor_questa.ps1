param([string]$QuestaRoot = 'D:\soft\questa\win64')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = $QuestaRoot
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_dit_qlinear_descriptor\questa_runs\$runTag"
$layout = Join-Path $repo 'build\supra2_dit_ddr_image\supra2_dit_ddr_layout.sv'

foreach ($tool in @('vlib.exe', 'vlog.exe', 'vsim.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $questa $tool))) {
        throw "Questa executable not found: $(Join-Path $questa $tool)"
    }
}
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
if (-not (Test-Path -LiteralPath $layout)) { throw "DDR layout package not found: $layout" }

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
        (Join-Path $repo 'rtl\supra2\supra2_dit_qlinear_descriptor.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_dit_qlinear_descriptor.sv')
    )
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $workDir 'qlinear_descriptor_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_dit_qlinear_descriptor -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
            -Pattern 'SUPRA2_DIT_QLINEAR_DESCRIPTOR_PASS layers=100 x_embed=576x16 ctx_proj=128x768->576' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    if (Select-String -LiteralPath $logPath -Pattern '\*\* Error:' -Quiet) {
        throw "Questa reported an error (see $logPath)"
    }
    if (Select-String -LiteralPath $logPath `
            -Pattern '(?m)^# (\*\* Warning:|Warnings: [1-9])' -Quiet) {
        throw "Questa reported an unsuppressed warning (see $logPath)"
    }
    "Questa all-layer QLinear descriptor simulation PASS; log: $logPath"
} finally {
    Pop-Location
}
