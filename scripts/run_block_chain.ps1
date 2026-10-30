param(
    [string]$ModelSimRoot = 'D:\soft\modelsim',
    [ValidateRange(0,13)][int]$StartBlock = 0,
    [ValidateRange(0,13)][int]$EndBlock = 13,
    [string]$InputFile = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ($StartBlock -gt $EndBlock) { throw 'StartBlock must not exceed EndBlock' }
if ($StartBlock -gt 0 -and -not $InputFile) {
    throw 'InputFile is required when starting after block 0'
}
if ($InputFile -and -not (Test-Path $InputFile)) {
    throw "Block-chain input does not exist: $InputFile"
}

Push-Location $repoRoot
try {
    $nextInput = $InputFile
    for ($block = $StartBlock; $block -le $EndBlock; $block++) {
        Write-Host "=== Supra2 RTL block $block/$EndBlock ==="
        $selfArgs = @{ ModelSimRoot = $ModelSimRoot; Block = $block }
        if ($nextInput) { $selfArgs.InputFile = $nextInput }
        & (Join-Path $PSScriptRoot 'run_layernorm_qkv_regression.ps1') @selfArgs
        if (-not $?) { throw "Self-attention regression failed for block $block" }

        $tag = if ($block -eq 0) { 'supra2' } else { "supra2_b$block" }
        if ($block -eq 0 -and -not $nextInput) {
            $blockInput = 'build/supra2_layernorm_modulate/x_fp16.hex'
        } else {
            $blockInput = $nextInput
        }
        if (-not $blockInput) { throw "Could not resolve original input to block $block" }

        $crossArgs = @{ ModelSimRoot = $ModelSimRoot; Block = $block;
                        InputFile = $blockInput }
        & (Join-Path $PSScriptRoot 'run_cross_attention_regression.ps1') @crossArgs
        if (-not $?) { throw "Cross-attention/MLP regression failed for block $block" }

        $outputFile = "build/${tag}_block${block}_mlp_residual/rtl_block${block}_output_fp16.hex"
        if (-not (Test-Path $outputFile)) { throw "Block $block did not emit $outputFile" }
        $outputBytes = (Get-Item -LiteralPath $outputFile).Length
        if ($outputBytes -ne 256 * 576 * 6) {
            throw "Block $block output has $outputBytes bytes; expected 884736"
        }
        $nextInput = $outputFile
        Write-Host "Block $block RTL output ready: $outputFile"
    }
    Write-Host "RTL block chain complete: blocks $StartBlock..$EndBlock; final tensor $nextInput"
}
finally {
    Pop-Location
}
