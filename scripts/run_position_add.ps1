param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [string]$LicenseFile = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$license = if ($LicenseFile) { $LicenseFile } else { Join-Path $ModelSimRoot 'LICENSE.TXT' }
$vlib = Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog = Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim = Join-Path $ModelSimRoot 'win64\vsim.exe'
$work = 'build/modelsim/work_supra2_position_add'
$outputDir = 'build/supra2_position_add'

foreach ($path in @($InputFile, $license, $vlib, $vlog, $vsim)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required file not found: $path" }
}

Push-Location $repoRoot
try {
    python -m compiler.supra2.prepare_position_add --input-file $InputFile --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw 'Position-add fixture generation failed' }
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license
    if (-not (Test-Path $work)) {
        & $vlib $work
        if ($LASTEXITCODE -ne 0) { throw 'vlib failed for position-add regression' }
    }
    & $vlog -sv -work $work -f rtl/supra2/position_add.f
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed for position-add regression' }
    $log = & $vsim -c -work $work tb_supra2_position_add `
        "+TOKEN_FILE=$outputDir/token_fp16.hex" `
        "+POSITION_FILE=$outputDir/position_fp16.hex" `
        "+EXPECTED_FILE=$outputDir/expected_fp16.hex" `
        "+ACTUAL_FILE=$outputDir/rtl_output_fp16.hex" `
        -do 'run -all; quit -f' 2>&1
    $log | Tee-Object -FilePath "$outputDir/modelsim.log"
    $logText = $log -join "`n"
    if ($LASTEXITCODE -ne 0 -or $logText -notmatch 'SUPRA2_POSITION_ADD_PASS elements=147456') {
        throw "Position-add RTL regression failed; see $outputDir/modelsim.log"
    }
}
finally {
    Pop-Location
}
