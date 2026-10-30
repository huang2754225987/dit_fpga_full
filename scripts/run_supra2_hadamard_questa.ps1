$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa = 'D:\soft\questa\win64'
$env:LM_LICENSE_FILE = 'C:\ProgramData\MentorGraphics\License\license.dat'
$env:MGLS_LICENSE_FILE = $env:LM_LICENSE_FILE
$runTag = Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir = Join-Path $repo "build\supra2_hadamard\questa_runs\$runTag"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$cases = @(
    @{ Name='x_embed'; K=16; H=16; Dir='supra2_x_embed_qlinear' },
    @{ Name='h16_two_blocks'; K=32; H=16; Dir='supra2_x_embed_qlinear' },
    @{ Name='h16_partial_group'; K=48; H=16; Dir='supra2_x_embed_qlinear' },
    @{ Name='qkv'; K=576; H=64; Dir='supra2_qkv_projection_fp32_frontend' },
    @{ Name='mlp2'; K=2304; H=256; Dir='supra2_mlp2_projection_fp32_frontend' }
)
foreach ($case in $cases) {
    $fixture = Join-Path $repo "build\$($case.Dir)"
    # For H=16, concatenate distinct archived patch rows to test block order.
    $inputWords = @(Get-Content (Join-Path $fixture 'raw_input_fp16.hex') -TotalCount $case.K)
    $expectedWords = @(Get-Content (Join-Path $fixture 'rotated_fp16.hex') -TotalCount $case.K)
    if ($inputWords.Count -ne $case.K -or $expectedWords.Count -ne $case.K) {
        throw "Incomplete existing frontend fixture for $($case.Name)"
    }
    $inputWords | Set-Content (Join-Path $workDir "$($case.Name)_input.hex")
    $expectedWords | Set-Content (Join-Path $workDir "$($case.Name)_expected.hex")
}
Push-Location $workDir
try {
    & (Join-Path $questa 'vlib.exe') work
    if ($LASTEXITCODE -ne 0) { throw 'vlib failed' }
    & (Join-Path $questa 'vlog.exe') -sv -work work `
        (Join-Path $repo 'rtl\pkg\fp16_utils_pkg.sv') `
        (Join-Path $repo 'rtl\supra2\fp32_add_rne_pipeline.sv') `
        (Join-Path $repo 'rtl\supra2\hadamard_transform.sv') `
        (Join-Path $repo 'rtl\sim\tb_supra2_hadamard.sv')
    if ($LASTEXITCODE -ne 0) { throw 'vlog failed' }
    foreach ($case in $cases) {
        $sim = & (Join-Path $questa 'vsim.exe') -c -lib work tb_supra2_hadamard `
            "+K=$($case.K)" "+H=$($case.H)" "+INPUT_FILE=$($case.Name)_input.hex" `
            "+EXPECTED_FILE=$($case.Name)_expected.hex" "+PACKAGE_FILE=$($case.Name)_expected.hex" `
            -do 'run -all; quit -f' 2>&1
        $simExit = $LASTEXITCODE
        $sim | Set-Content "$($case.Name)_questa.log"
        $sim
        $simText = $sim -join "`n"
        if ($simExit -ne 0 -or $simText -notmatch "SUPRA2_HADAMARD_PASS K=$($case.K) H=$($case.H)" -or
            $simText -notmatch 'SUPRA2_HADAMARD_BACKPRESSURE_PASS' -or
            $simText -match '\*\* (Fatal|Error)|Errors: [1-9]') {
            throw "Hadamard regression failed for $($case.Name)"
        }
    }
    "Questa Hadamard golden regression PASS; logs: $workDir"
} finally { Pop-Location }
