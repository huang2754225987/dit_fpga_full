param(
    [string]$ModelSimRoot = 'D:\soft\modelsim'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vlib = Join-Path $ModelSimRoot 'win64\vlib.exe'
$vlog = Join-Path $ModelSimRoot 'win64\vlog.exe'
$vsim = Join-Path $ModelSimRoot 'win64\vsim.exe'
$license = Join-Path $ModelSimRoot 'LICENSE.TXT'
$workLibrary = 'build/modelsim/work_supra2_euler'
$outputDir = 'build/supra2_euler_replay'

foreach ($tool in @($vlib, $vlog, $vsim, $license)) {
    if (-not (Test-Path $tool)) { throw "Required ModelSim file not found: $tool" }
}

Push-Location $repoRoot
try {
    $env:LM_LICENSE_FILE = $license
    $env:MGLS_LICENSE_FILE = $license
    python -m compiler.supra2.prepare_euler_replay --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw 'Euler fixture generation failed' }

    if (-not (Test-Path $workLibrary)) {
        & $vlib $workLibrary
        if ($LASTEXITCODE -ne 0) { throw "vlib failed with exit code $LASTEXITCODE" }
    }
    & $vlog -sv -work $workLibrary -f rtl/supra2/euler_replay.f
    if ($LASTEXITCODE -ne 0) { throw "vlog failed with exit code $LASTEXITCODE" }

    $simArgs = @(
        '-c', '-work', $workLibrary, 'tb_supra2_euler_replay',
        "+Z0_FILE=$outputDir/z0_fp16.hex",
        "+VELOCITY_FILE=$outputDir/velocity_steps_fp16.hex",
        "+EXPECTED_FILE=$outputDir/z_final_fp16.hex",
        "+REFERENCE_FILE=$outputDir/reference_final_fp16.hex",
        '-do', 'run -all; quit -f'
    )
    $simOutput = & $vsim @simArgs 2>&1
    $simOutput | Tee-Object -FilePath "$outputDir/modelsim.log"
    $simText = $simOutput -join "`n"
    if ($LASTEXITCODE -ne 0 -or
        $simText -notmatch 'SUPRA2_EULER_REPLAY_PASS steps=50 elements=4096' -or
        $simText -notmatch 'SUPRA2_EULER_REPLAY_SOFTWARE_MISMATCHES 0' -or
        $simText -match '\*\* Fatal|Errors: [1-9]') {
        throw "Euler RTL replay failed; see $outputDir/modelsim.log"
    }
}
finally {
    Pop-Location
}
