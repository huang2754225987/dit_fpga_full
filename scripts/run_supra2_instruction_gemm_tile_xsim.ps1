param([string]$OutputDirectory='')
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if(!$OutputDirectory){
    $tag=Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDirectory=Join-Path $repo "build\supra2_instruction_gemm_tile\$tag"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path $OutputDirectory).Path
Push-Location $repo
try{
    & python -m compiler.supra2.compile_block13_tile_program --output-dir build/supra2_tile_program | Out-Null
    if($LASTEXITCODE -ne 0){throw 'Tile program compilation failed'}
    & python -m compiler.supra2.extract_single_tile_instruction `
        --program-json build/supra2_tile_program/block13_tile_program.json `
        --output-hex (Join-Path $OutputDirectory 'qkv_row0_instruction.hex') `
        --output-n-tiles 1 | Out-Null
    if($LASTEXITCODE -ne 0){throw 'Single GEMM_TILE extraction failed'}
    $instruction=Get-Content (Join-Path $OutputDirectory 'qkv_row0_instruction.json') -Raw | ConvertFrom-Json
    $fixture=Join-Path $repo 'build\supra2_qkv_projection_fp32_frontend'
    foreach($file in @('raw_input_fp16.hex','weight_int8.hex','weight_scale_fp32.hex',
        'bias_fp32.hex','reference_output_fp16.hex')){
        if(!(Test-Path -LiteralPath (Join-Path $fixture $file))){
            throw "Missing QKV single-tile golden fixture: $fixture\$file"
        }
    }
    $referenceLines=[System.IO.File]::ReadAllLines((Join-Path $fixture 'reference_output_fp16.hex'))
    if($referenceLines.Count -lt (256*1728)){throw 'Incomplete block-13 QKV reference tensor'}
    $expectedPath=Join-Path $OutputDirectory 'qkv_first_n_tile_expected.hex'
    $expectedLines=[System.Collections.Generic.List[string]]::new()
    for($row=0;$row -lt 32;$row++){
        for($lane=0;$lane -lt 32;$lane++){$expectedLines.Add($referenceLines[$row*1728+$lane])}
    }
    [System.IO.File]::WriteAllLines($expectedPath,$expectedLines)
    $stats=Join-Path $OutputDirectory 'instruction_stats.csv'
    $plusargs=@(
        ('PROGRAM_FILE='+(Join-Path $OutputDirectory 'qkv_row0_instruction.hex').Replace('\','/')),
        ('RAW='+(Join-Path $fixture 'raw_input_fp16.hex').Replace('\','/')),
        ('WEIGHT='+(Join-Path $fixture 'weight_int8.hex').Replace('\','/')),
        ('SCALE='+(Join-Path $fixture 'weight_scale_fp32.hex').Replace('\','/')),
        ('BIAS='+(Join-Path $fixture 'bias_fp32.hex').Replace('\','/')),
        ('EXPECTED='+$expectedPath.Replace('\','/')),
        ('STATS='+$stats.Replace('\','/')),
        ('SOURCE_PC='+$instruction.source_pc)
    )
    $tag=Get-Date -Format 'yyyyMMdd_HHmmss'
    & scripts/run_xsim_standalone.ps1 -FileList rtl/supra2/gemm_tile_instruction.f `
        -Top tb_supra2_instruction_gemm_tile -Snapshot "supra2_gemm_tile_instruction_$tag" `
        -PassPattern "SUPRA2_TILE_GEMM_PASS source_pc=$($instruction.source_pc) stage=QKV row_tile=0 M=32 N=32 K=576" `
        -DebugLevel off -TestPlusArgs $plusargs
    if($LASTEXITCODE -ne 0){throw 'Instruction-driven GEMM tile simulation failed'}
    Write-Output "TILE_INSTRUCTION_STATS=$stats"
}finally{Pop-Location}
