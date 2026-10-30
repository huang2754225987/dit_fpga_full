param(
    [string]$InputFile = 'build/supra2_z0_layernorm/rtl_modulated_fp16.hex',
    [string]$BatchRoot = 'build/supra2_z0_qkv_batches',
    [int]$RowsPerBatch = 32,
    [string]$VivadoRoot = 'D:\soft\vitis\Vivado\2021.1',
    [string]$Msys2Root = 'D:\C\msys2\ucrt64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($RowsPerBatch -ne 32) { throw 'The current QLinear regression requires 32-row M tiles' }
Push-Location $repoRoot
try {
    $batchCount = 256 / $RowsPerBatch
    $fullDir = Join-Path $BatchRoot 'full_fixture'
    python -m compiler.supra2.prepare_qlinear_projection `
        --layer blocks.0.self_attn.qkv --fp16-hadamard-frontend `
        --raw-input-file $InputFile --output-dir $fullDir
    if ($LASTEXITCODE -ne 0) { throw 'Full QKV reference fixture generation failed' }

    $snapshot = 'supra2_z0_qkv_batch_v2'
    for ($batchIndex = 0; $batchIndex -lt $batchCount; $batchIndex++) {
        $firstRow = $batchIndex * $RowsPerBatch
        $batchName = 'batch_{0:D2}' -f $batchIndex
        $batchDir = Join-Path $BatchRoot $batchName
        $actualFile = Join-Path $batchDir 'rtl_output_fp16.hex'
        $batchLog = Join-Path $batchDir 'xsim.log'
        $cycleFile = Join-Path $batchDir 'rtl_cycles.txt'

        if ((Test-Path $actualFile) -and (Test-Path $batchLog) -and
            (Select-String -Path $batchLog -Pattern "SUPRA2_QPROJ_PASS M=32 N=1728 K=576" -Quiet)) {
            Write-Host "QKV batch $batchIndex already passed; reusing its RTL tensor."
            continue
        }

        python -m compiler.supra2.prepare_qlinear_projection `
            --layer blocks.0.self_attn.qkv --fp16-hadamard-frontend `
            --raw-input-file $InputFile --row-start $firstRow `
            --row-count $RowsPerBatch --output-dir $batchDir
        if ($LASTEXITCODE -ne 0) { throw "QKV fixture generation failed for batch $batchIndex" }

        $runnerParams = @{
            FileList = 'rtl/supra2/qlinear_projection.f'
            Top = 'tb_supra2_qlinear_projection'
            Snapshot = $snapshot
            PassPattern = 'SUPRA2_QPROJ_PASS M=32 N=1728 K=576'
            VivadoRoot = $VivadoRoot
            Msys2Root = $Msys2Root
            TestPlusArgs = @(
                'M=32', 'N=1728', 'K=576', 'H=64',
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
        if ($batchIndex -eq 0 -and -not (Test-Path "xsim.dir/$snapshot/xsimk.exe")) {
            $runnerParams['Defines'] = @('SUPRA2_XSIM_QKV_BATCH')
        }
        else {
            $runnerParams['ReuseSnapshot'] = $true
        }

        & (Join-Path $PSScriptRoot 'run_xsim_standalone.ps1') @runnerParams
        if (-not $?) { throw "XSim failed for QKV batch $batchIndex" }
        Copy-Item "build/xsim_$snapshot.sim.log" $batchLog -Force
        $cycleMatch = Select-String -Path $batchLog -Pattern 'SUPRA2_QPROJ_CYCLES (\d+)'
        if (-not $cycleMatch) { throw "Cycle count missing for QKV batch $batchIndex" }
        $cycleMatch.Matches[0].Groups[1].Value |
            Set-Content -Path $cycleFile -NoNewline
        Write-Host "QKV batch $batchIndex/$($batchCount-1) passed."
    }

    $totalCycles = 0
    for ($batchIndex = 0; $batchIndex -lt $batchCount; $batchIndex++) {
        $cycleFile = Join-Path (Join-Path $BatchRoot ('batch_{0:D2}' -f $batchIndex)) 'rtl_cycles.txt'
        if (-not (Test-Path $cycleFile)) {
            throw "Missing verified QKV batch cycle record: $cycleFile"
        }
        $totalCycles += [int](Get-Content $cycleFile -Raw)
    }
    python -m compiler.supra2.merge_qlinear_batches `
        --batch-root $BatchRoot --batch-count $batchCount `
        --rows-per-batch $RowsPerBatch --features 1728 `
        --expected-file "$fullDir/reference_output_fp16.hex" `
        --output-file "$BatchRoot/rtl_output_fp16.hex" --cycles $totalCycles
    if ($LASTEXITCODE -ne 0) { throw 'Merged QKV RTL tensor did not match the full reference' }
}
finally {
    Pop-Location
}
