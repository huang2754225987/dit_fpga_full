$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_dit_final_param_dma\questa_runs\$runTag"

foreach ($tool in @('vlib.exe', 'vlog.exe', 'vsim.exe')) {
    if (-not (Test-Path -LiteralPath (Join-Path $questa $tool))) {
        throw "Questa executable not found: $(Join-Path $questa $tool)"
    }
}
if (-not (Test-Path -LiteralPath $license)) { throw "Questa license not found: $license" }
$layout = Join-Path $repo 'build\supra2_dit_ddr_image\supra2_dit_ddr_layout.sv'
if (-not (Test-Path -LiteralPath $layout)) { throw "Missing DDR layout: $layout" }
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
        (Join-Path $repo 'rtl\supra2\supra2_axi_read_dma.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dit_runtime_controller.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dicache_online_policy.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_dit_runtime_dma.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_dit_final_param_dma.sv')
    )
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $workDir 'final_param_dma_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -c -lib work `
        tb_supra2_dit_final_param_dma -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
            -Pattern 'SUPRA2_FINAL_PARAM_DMA_PASS weight=288 bias=1 adaln=72 bursts=8 cached_static=1' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    "Questa final-param DMA simulation PASS; log: $logPath"
} finally { Pop-Location }
