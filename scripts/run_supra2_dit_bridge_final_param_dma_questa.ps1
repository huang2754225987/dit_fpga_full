$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_bridge_final_param_dma\questa_runs\$runTag"
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
$lutDir = Join-Path $workDir 'rtl\supra2'
New-Item -ItemType Directory -Path $lutDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'rtl\supra2\gelu_tanh_lut_fp16.hex') -Destination $lutDir
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    $sources = @(Get-Content (Join-Path $repo 'rtl\supra2\supra2_dit_compute_bridge.f') |
        Where-Object { $_ -and -not $_.StartsWith('#') } |
        ForEach-Object { Join-Path $repo ($_ -replace '/', '\') })
    $sources += Join-Path $repo 'rtl\sim\tb_supra2_dit_bridge_final_param_dma.sv'
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }
    $logPath = Join-Path $workDir 'bridge_final_param_dma_questa.log'
    $simOutput = & (Join-Path $questa 'vsim.exe') -suppress 7061 -c -lib work `
        tb_supra2_dit_bridge_final_param_dma -do 'run -all; quit -f' 2>&1
    $simExit = $LASTEXITCODE
    $simOutput | Set-Content -LiteralPath $logPath
    $simOutput
    if ($simExit -ne 0) { throw "vsim failed with exit code $simExit (see $logPath)" }
    if (-not (Select-String -LiteralPath $logPath `
            -Pattern 'SUPRA2_BRIDGE_FINAL_PARAM_DMA_PASS qlinear_beats=16 vector_beats=7 weight=288 bias=16 shift=18 scale=18 output=128' -Quiet)) {
        throw "Questa completed without the required PASS marker (see $logPath)"
    }
    "Questa bridge final-param DMA simulation PASS; log: $logPath"
} finally { Pop-Location }
