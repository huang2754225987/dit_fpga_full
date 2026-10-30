param([string]$OutputDirectory = '')
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (!$OutputDirectory) { $OutputDirectory = Join-Path $repo 'build\supra2_tile_program' }
Push-Location $repo
try {
    & python -m compiler.supra2.compile_block13_tile_program --output-dir $OutputDirectory | Out-Null
    if ($LASTEXITCODE) { throw 'Tile program compilation failed' }
    $report = Get-Content -LiteralPath (Join-Path $OutputDirectory 'block13_tile_schedule.json') -Raw | ConvertFrom-Json
    if ($report.rtl_replay_event_count -gt 2048) { throw 'RTL replay task limit exceeded' }
    $table = (Resolve-Path (Join-Path $OutputDirectory 'block13_tile_schedule.hex')).Path.Replace('\','/')
    $program = (Resolve-Path (Join-Path $OutputDirectory 'block13_tile_program.hex')).Path.Replace('\','/')
    $dependencies = (Resolve-Path (Join-Path $OutputDirectory 'block13_tile_dependencies.hex')).Path.Replace('\','/')
    $programManifest = Get-Content -LiteralPath (Join-Path $OutputDirectory 'block13_tile_program.json') -Raw | ConvertFrom-Json
    $tag = Get-Date -Format 'yyyyMMdd_HHmmss'
    $args = @(
        ('NTASK=' + $report.rtl_replay_event_count),
        ('EXPECTED_MAKESPAN=' + $report.makespan_cycles),
        ('TABLE_FILE=' + $table),
        ('PROGRAM_FILE=' + $program),
        ('DEPENDENCY_FILE=' + $dependencies),
        ('DEPENDENCY_WORDS=' + $programManifest.dependency_word_count),
        ('EXPECTED_INSTRUCTIONS=' + $programManifest.issue_count)
    )
    & scripts/run_xsim_standalone.ps1 -FileList rtl/supra2/compiled_tile_schedule.f `
        -Top tb_supra2_compiled_tile_schedule -Snapshot "supra2_tile_schedule_$tag" `
        -PassPattern "SUPRA2_COMPILED_TILE_SCHEDULE_PASS makespan=$($report.makespan_cycles) events=$($report.rtl_replay_event_count)" `
        -DebugLevel off -TestPlusArgs $args
}
finally { Pop-Location }
