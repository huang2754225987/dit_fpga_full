param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$VivadoRoot = 'D:\soft\vitis\Vivado\2021.1',
    [string]$Msys2Root = 'D:\C\msys2\ucrt64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    $outputDir = 'build/supra2_position_add'
    python -m compiler.supra2.prepare_position_add --input-file $InputFile --output-dir $outputDir
    if ($LASTEXITCODE -ne 0) { throw 'Position-add fixture generation failed' }

    & (Join-Path $PSScriptRoot 'run_xsim_standalone.ps1') `
        -FileList 'rtl/supra2/position_add.f' `
        -Top 'tb_supra2_position_add' `
        -Snapshot 'supra2_position_add' `
        -PassPattern 'SUPRA2_POSITION_ADD_PASS elements=147456' `
        -VivadoRoot $VivadoRoot -Msys2Root $Msys2Root `
        -TestPlusArgs @(
            "TOKEN_FILE=$outputDir/token_fp16.hex",
            "POSITION_FILE=$outputDir/position_fp16.hex",
            "EXPECTED_FILE=$outputDir/expected_fp16.hex",
            "ACTUAL_FILE=$outputDir/rtl_output_fp16.hex"
        )
    if (-not $?) { throw 'Position-add standalone XSim regression failed' }
}
finally {
    Pop-Location
}
