$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_axi_read_dma\questa_runs\$runTag"

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
        (Join-Path $repo 'rtl\supra2\supra2_axi_read_dma.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_axi_read_dma.sv')
    )
    & (Join-Path $questa 'vlog.exe') -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }

    $logPath = Join-Path $workDir 'axi_read_dma_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_axi_read_dma -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
        -Pattern 'SUPRA2_AXI_READ_DMA_PASS boundary_split=1\+3 tile_beats=16 max_burst=64 max_descriptor=2\^26' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    "Questa AXI read DMA simulation PASS; log: $logPath"
} finally {
    Pop-Location
}
