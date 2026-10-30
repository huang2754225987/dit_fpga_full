param(
    [Parameter(Mandatory=$true)][string]$Layer,
    [Parameter(Mandatory=$true)][string]$RawInputFile,
    [Parameter(Mandatory=$true)][int]$Rows,
    [Parameter(Mandatory=$true)][int]$Features,
    [Parameter(Mandatory=$true)][int]$N,
    [Parameter(Mandatory=$true)][int]$K,
    [Parameter(Mandatory=$true)][int]$HadamardBlock,
    [Parameter(Mandatory=$true)][string]$BatchRoot,
    [ValidateRange(1,32)][int]$RowsPerBatch = 32,
    [string]$VivadoRoot = 'D:\soft\vitis\Vivado\2021.1',
    [string]$Msys2Root = 'D:\C\msys2\ucrt64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($RowsPerBatch -ne 32) { throw 'Current QLinear array schedule requires 32-row batches' }
if ($Rows -le 0 -or $Rows % $RowsPerBatch -ne 0) {
    throw 'Rows must be a positive multiple of RowsPerBatch'
}
if (-not (Test-Path -LiteralPath $RawInputFile)) { throw "Input tensor not found: $RawInputFile" }

Push-Location $repoRoot
try {
    $fullDir = Join-Path $BatchRoot 'full_fixture'
    python -m compiler.supra2.prepare_qlinear_projection `
        --layer $Layer --fp16-hadamard-frontend --raw-input-file $RawInputFile `
        --output-dir $fullDir
    if ($LASTEXITCODE -ne 0) { throw 'Full QLinear reference fixture generation failed' }

    $batchCount = $Rows / $RowsPerBatch
    $safeLayer = $Layer -replace '[^A-Za-z0-9]+', '_'
    $snapshot = "supra2_${safeLayer}_xsim"
    for ($batchIndex = 0; $batchIndex -lt $batchCount; $batchIndex++) {
        $firstRow = $batchIndex * $RowsPerBatch
        $batchName = 'batch_{0:D2}' -f $batchIndex
        $batchDir = Join-Path $BatchRoot $batchName
        $actualFile = Join-Path $batchDir 'rtl_output_fp16.hex'
        $batchLog = Join-Path $batchDir 'xsim.log'
        $cycleFile = Join-Path $batchDir 'rtl_cycles.txt'
        $passPattern = "SUPRA2_QPROJ_PASS M=$RowsPerBatch N=$N K=$K"

        if ((Test-Path $actualFile) -and (Test-Path $batchLog) -and
            (Select-String -Path $batchLog -SimpleMatch $passPattern -Quiet)) {
            Write-Host "QLinear batch $batchIndex already passed; reusing its RTL tensor."
            if (-not (Test-Path $cycleFile)) {
                $cycleMatch = Select-String -Path $batchLog -Pattern 'SUPRA2_QPROJ_CYCLES (\d+)'
                if (-not $cycleMatch) { throw "Cycle count missing from $batchLog" }
                $cycleMatch.Matches[0].Groups[1].Value |
                    Set-Content -Path $cycleFile -NoNewline
            }
            continue
        }

        python -m compiler.supra2.prepare_qlinear_projection `
            --layer $Layer --fp16-hadamard-frontend --raw-input-file $RawInputFile `
            --row-start $firstRow --row-count $RowsPerBatch --output-dir $batchDir
        if ($LASTEXITCODE -ne 0) { throw "Fixture generation failed for batch $batchIndex" }

        $runnerParams = @{
            FileList = 'rtl/supra2/qlinear_projection.f'
            Top = 'tb_supra2_qlinear_projection'
            Snapshot = $snapshot
            PassPattern = $passPattern
            VivadoRoot = $VivadoRoot
            Msys2Root = $Msys2Root
            DebugLevel = 'off'
            TestPlusArgs = @(
                "M=$RowsPerBatch", "N=$N", "K=$K", "H=$HadamardBlock",
                "RAW_INPUT_FILE=$batchDir/raw_input_fp16.hex",
                "EXPECTED_Q_FILE=$batchDir/expected_quantized_int8.hex",
                "EXPECTED_ROTATED_FILE=$batchDir/rotated_fp16.hex",
                "WEIGHT_FILE=$batchDir/weight_int8.hex",
                "ACC_FILE=$batchDir/accumulator_int32.hex",
                "OUTPUT_FILE=$batchDir/output_fp16.hex",
                "ACTUAL_OUTPUT_FILE=$actualFile",
                "REFERENCE_OUTPUT_FILE=$batchDir/reference_output_fp16.hex",
                "TOKEN_SCALE_FILE=$batchDir/token_scale_fp32.hex",
                "WEIGHT_SCALE_FILE=$batchDir/weight_scale_fp32.hex",
                "BIAS_FILE=$batchDir/bias_fp32.hex"
            )
        }
        if ($batchIndex -gt 0 -or (Test-Path "xsim.dir/$snapshot/xsimk.exe")) {
            $runnerParams['ReuseSnapshot'] = $true
        }
        & (Join-Path $PSScriptRoot 'run_xsim_standalone.ps1') @runnerParams
        if (-not $?) { throw "XSim failed for QLinear batch $batchIndex" }
        Copy-Item "build/xsim_$snapshot.sim.log" $batchLog -Force
        $cycleMatch = Select-String -Path $batchLog -Pattern 'SUPRA2_QPROJ_CYCLES (\d+)'
        if (-not $cycleMatch) { throw "Cycle count missing for batch $batchIndex" }
        $cycleMatch.Matches[0].Groups[1].Value |
            Set-Content -Path $cycleFile -NoNewline
        Write-Host "QLinear batch $batchIndex/$($batchCount-1) passed."
    }

    $totalCycles = 0
    for ($batchIndex = 0; $batchIndex -lt $batchCount; $batchIndex++) {
        $cycleFile = Join-Path (Join-Path $BatchRoot ('batch_{0:D2}' -f $batchIndex)) 'rtl_cycles.txt'
        if (-not (Test-Path $cycleFile)) { throw "Missing verified cycle record: $cycleFile" }
        $totalCycles += [int](Get-Content $cycleFile -Raw)
    }
    python -m compiler.supra2.merge_qlinear_batches `
        --batch-root $BatchRoot --batch-count $batchCount `
        --rows-per-batch $RowsPerBatch --features $N `
        --expected-file "$fullDir/reference_output_fp16.hex" `
        --output-file "$BatchRoot/rtl_output_fp16.hex" --cycles $totalCycles
    if ($LASTEXITCODE -ne 0) { throw 'Merged QLinear tensor did not match the full reference' }
}
finally {
    Pop-Location
}
