param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [ValidateRange(0,13)][int]$EndBlock = 13
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $repoRoot
try {
    & (Join-Path $PSScriptRoot 'run_qlinear_frontend_regression.ps1') `
        -ModelSimRoot $ModelSimRoot -Only x_embed
    if (-not $?) { throw 'Archive-z0 x_embed RTL regression failed' }

    $xEmbedOutput = 'build/supra2_x_embed_qlinear/rtl_output_fp16.hex'
    if (-not (Test-Path -LiteralPath $xEmbedOutput)) {
        throw "x_embed did not produce $xEmbedOutput"
    }

    & (Join-Path $PSScriptRoot 'run_position_add.ps1') `
        -InputFile $xEmbedOutput -ModelSimRoot $ModelSimRoot
    if (-not $?) { throw 'Position-embedding RTL regression failed' }

    $block0Input = 'build/supra2_position_add/rtl_output_fp16.hex'
    if (-not (Test-Path -LiteralPath $block0Input)) {
        throw "Position add did not produce $block0Input"
    }

    & (Join-Path $PSScriptRoot 'run_block_chain.ps1') `
        -ModelSimRoot $ModelSimRoot -StartBlock 0 -EndBlock $EndBlock `
        -InputFile $block0Input
    if (-not $?) { throw 'RTL block chain failed' }

    Write-Host "z0 -> x_embed -> pos_embed -> blocks 0..$EndBlock complete (single t=0 branch only)."
}
finally {
    Pop-Location
}
