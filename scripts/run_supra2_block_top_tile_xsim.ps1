param([string]$OutputDirectory='')
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if(!$OutputDirectory){
    $tag=Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputDirectory=Join-Path $repo "build\supra2_block13_tile_xsim\$tag"
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory=(Resolve-Path $OutputDirectory).Path
Push-Location $repo
try{
    & python -m compiler.supra2.compile_block13_tile_program --output-dir build/supra2_tile_program | Out-Null
    if($LASTEXITCODE -ne 0){throw 'Tile program compilation failed'}
    $program=Get-Content 'build/supra2_tile_program/block13_tile_program.json' -Raw | ConvertFrom-Json
    $report=Get-Content 'build/supra2_tile_program/block13_compile_report.json' -Raw | ConvertFrom-Json
    $fixture=Join-Path $repo 'build\supra2_block13_top'
    foreach($file in @('x_input.hex','ctx_packed.hex','golden_y.hex',
        'ln0_scale.hex','ln0_shift.hex','ln1_scale.hex','ln1_shift.hex',
        'ln2_scale.hex','ln2_shift.hex','gate0.hex','gate1.hex')){
        if(!(Test-Path -LiteralPath (Join-Path $fixture $file))){
            throw "Missing block-13 fixture: $fixture\$file"
        }
    }
    $tag=Get-Date -Format 'yyyyMMdd_HHmmss'
    $stats=Join-Path $OutputDirectory 'tile_instruction_stats.csv'
    $enriched=Join-Path $OutputDirectory 'tile_instruction_stats_enriched.csv'
    $heartbeat=Join-Path $OutputDirectory 'tile_heartbeat.log'
    $actual=Join-Path $OutputDirectory 'rtl_actual_tile_fp16.hex'
    $plusargs=@(
        ('DIR='+$fixture.Replace('\','/')),
        ('ACTUAL_FILE='+$actual.Replace('\','/')),
        ('TILE_STATS='+$stats.Replace('\','/')),
        ('TILE_HEARTBEAT='+$heartbeat.Replace('\','/')),
        ('TILE_PROGRAM_WORDS='+$program.issue_count),
        ('TILE_DEPENDENCY_WORDS='+$program.dependency_word_count),
        'BLOCK_ID=13'
    )
    & scripts/run_xsim_standalone.ps1 -FileList rtl/supra2/supra2_block_top.f `
        -Top tb_supra2_block_top -TopGenerics @('USE_TILE_PROGRAM=1') `
        -Snapshot "supra2_block13_tile_$tag" -PassPattern ("SUPRA2_TILE_PROGRAM_PASS commands=$($report.schedule_tasks)") `
        -DebugLevel off -StreamOutput -TestPlusArgs $plusargs
    if($LASTEXITCODE -ne 0){throw 'Instruction-driven block simulation failed'}
    & python -m compiler.supra2.enrich_tile_instruction_stats `
        --program build/supra2_tile_program/block13_tile_program.json `
        --stats $stats --output $enriched
    if($LASTEXITCODE -ne 0){throw 'Tile instruction statistics enrichment failed'}
    Write-Output "TILE_INSTRUCTION_STATS=$stats"
    Write-Output "TILE_INSTRUCTION_STATS_ENRICHED=$enriched"
    Write-Output "TILE_HEARTBEAT=$heartbeat"
    Write-Output "ACTUAL_OUTPUT=$actual"
}finally{Pop-Location}
