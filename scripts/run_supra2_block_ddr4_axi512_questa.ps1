param(
    [string]$QuestaRoot = 'D:\soft\questa\win64',
    [string]$RunDirectory = '',
    [switch]$CalibrationOnly,
    [switch]$TileProgram,
    [switch]$CompileOnly
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$ipRun = Join-Path $repo 'build\board_mig_axi512_sim\ip_export\ddr4_pl_axi_8\questa'
$payload = Join-Path $repo 'build\supra2_block13_ddr_traffic\block13_payload.bin'
$layout = Join-Path $repo 'build\supra2_block13_ddr_traffic\block13_ddr_layout.sv'
$golden = Join-Path $repo 'build\supra2_block13_top\golden_y.hex'
$license = 'C:\ProgramData\MentorGraphics\License\license.dat'
$modelRoot = 'D:\soft\vitis\Vivado\2021.1\data\ip\xilinx\ddr4_v2_2\data\dlib\ultrascale\ddr4_sdram\tb\ddr4_model'
$modelInclude = Join-Path $repo 'rtl\sim\ddr4_axi512_model'
$modelWrapper = Join-Path $modelInclude 'ddr4_sdram_model_wrapper.sv'
$programDir = Join-Path $repo 'build\supra2_block_program'
$programSource = Join-Path $programDir 'block13_stage_program.hex'
$tileProgramDir = Join-Path $repo 'build\supra2_tile_program'
foreach ($file in @($payload, $layout, $golden, $license, $modelWrapper,
        (Join-Path $ipRun 'modelsim.ini'), (Join-Path $QuestaRoot 'vsim.exe'))) {
    if (!(Test-Path -LiteralPath $file)) { throw "Required simulation input missing: $file" }
}
if (!(Test-Path (Join-Path $ipRun 'questa_lib\msim\xil_defaultlib'))) {
    throw 'Compile the exported board MIG simulation library before running this script.'
}
$xci = Get-Content (Join-Path $repo 'build\zcu106_pl\ip_axi\ddr4_pl_axi_8\ddr4_pl_axi.xci') -Raw
if ($xci -notmatch 'PARAM_VALUE.C0.DDR4_AxiDataWidth">512<' -or
    $xci -notmatch 'PARAM_VALUE.C0.DDR4_DataWidth">64<') {
    throw 'This test requires the actual AXI512 / physical DQ64 board MIG configuration.'
}
Push-Location $repo
try {
    & python -m compiler.supra2.compile_block_program --output-dir $programDir | Out-Null
    if ($LASTEXITCODE) { throw 'Block stage program compilation failed' }
    if ($TileProgram) {
        & python -m compiler.supra2.compile_block13_tile_program --output-dir $tileProgramDir | Out-Null
        if ($LASTEXITCODE) { throw 'Block tile program compilation failed' }
    }
} finally {
    Pop-Location
}
if (!$RunDirectory) {
    $tag = Get-Date -Format 'yyyyMMdd_HHmmss'
    $RunDirectory = Join-Path $repo "build\board_mig_axi512_sim\runs\$tag"
}
New-Item -ItemType Directory -Force $RunDirectory | Out-Null
$RunDirectory = (Resolve-Path $RunDirectory).Path
# Vendor libraries are immutable shared inputs; application RTL gets a fresh
# local work library and modelsim.ini, never the previous simulation snapshot.
$ini = Get-Content (Join-Path $ipRun 'modelsim.ini') -Raw
$absoluteLibs = (Join-Path $ipRun 'questa_lib\msim').Replace('\','/') + '/'
$ini = $ini.Replace('= questa_lib/msim/', '= ' + $absoluteLibs)
Set-Content (Join-Path $RunDirectory 'modelsim.ini') $ini
$lutDir = Join-Path $RunDirectory 'rtl\supra2'
New-Item -ItemType Directory -Force $lutDir | Out-Null
Copy-Item (Join-Path $repo 'rtl\supra2\gelu_tanh_lut_fp16.hex') $lutDir
Copy-Item $programSource (Join-Path $RunDirectory 'block13_stage_program.hex')
if ($TileProgram) {
    Copy-Item (Join-Path $tileProgramDir 'block13_tile_program.hex') (Join-Path $RunDirectory 'block13_tile_program.hex')
    Copy-Item (Join-Path $tileProgramDir 'block13_tile_dependencies.hex') (Join-Path $RunDirectory 'block13_tile_dependencies.hex')
    Copy-Item (Join-Path $tileProgramDir 'block13_tile_program.bin') (Join-Path $RunDirectory 'block13_tile_program.bin')
    Copy-Item (Join-Path $tileProgramDir 'block13_tile_dependencies.bin') (Join-Path $RunDirectory 'block13_tile_dependencies.bin')
}
$env:LM_LICENSE_FILE = $license
$env:MGLS_LICENSE_FILE = $license
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
$env:PATH = "$QuestaRoot;$env:PATH"
Push-Location $RunDirectory
try {
    & (Join-Path $QuestaRoot 'vlib.exe') work
    if ($LASTEXITCODE) { throw 'vlib failed' }
    & (Join-Path $QuestaRoot 'vmap.exe') work (Join-Path $RunDirectory 'work')
    if ($LASTEXITCODE) { throw 'vmap failed' }
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work work -l ddr_model_compile.log `
        "+incdir+$modelInclude" "+incdir+$modelRoot" $modelWrapper
    if ($LASTEXITCODE) { throw 'DDR4 x16 device-model compilation failed' }
    $sources = @(Get-Content (Join-Path $repo 'rtl\supra2\supra2_block_top.f') |
        Where-Object { $_ -and !($_.StartsWith('#')) -and $_ -ne 'rtl/sim/tb_supra2_block_top.sv' } |
        ForEach-Object { Join-Path $repo $_ })
    $sources += @($layout,
        (Join-Path $repo 'rtl\supra2\supra2_block_stage_sequencer.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_tile_program_ddr_loader.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_block_dma.sv'),
        (Join-Path $repo 'rtl\supra2\supra2_block_ddr_system_top.sv'),
        (Join-Path $repo 'rtl\sim\tb_supra2_block_ddr4_axi512_system.sv'),
        (Join-Path $ipRun 'glbl.v'))
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work work -l rtl_compile.log @sources
    if ($LASTEXITCODE) { throw 'Connected AXI512 DDR system RTL compilation failed' }
    $manifest = [ordered]@{
        started = (Get-Date -Format o); axi_width = 512; ddr_dq_width = 64
        devices = '4 x x16 MT40A256M16GE-075E'; backend = 'Questa'
        calibration_only = [bool]$CalibrationOnly
        payload = $payload; payload_bytes = (Get-Item $payload).Length; golden = $golden
        stage_program = (Get-FileHash $programSource -Algorithm SHA256)
        tile_program = if ($TileProgram) { Get-FileHash (Join-Path $tileProgramDir 'block13_tile_program.hex') -Algorithm SHA256 } else { $null }
        sources = @($sources | ForEach-Object { Get-FileHash $_ -Algorithm SHA256 } |
            Select-Object Path,Hash)
    }
    $manifest | ConvertTo-Json -Depth 5 | Set-Content run_manifest.json
    if ($CompileOnly) {
        Write-Output "AXI512_DDR64_COMPILE_ONLY_PASS run_directory=$RunDirectory tile_program=$TileProgram"
        return
    }
    $simArgs = @('-c', '-suppress', '7061', '-t', '1ps', '-L', 'xil_defaultlib',
        '-L', 'unisims_ver', '-L', 'unimacro_ver', '-L', 'secureip', '-L', 'xpm',
        'work.tb_supra2_block_ddr4_axi512_system', 'work.glbl', '-l', 'system.log',
        ('+PAYLOAD=' + $payload.Replace('\','/')),
        ('+BYTES=' + (Get-Item $payload).Length),
        ('+GOLDEN=' + $golden.Replace('\','/')),
        '-do', 'onbreak {quit -code 1}; onerror {quit -code 1}; run -all; quit -code 0')
    if ($TileProgram) {
        $tileManifest = Get-Content (Join-Path $tileProgramDir 'block13_tile_program.json') -Raw | ConvertFrom-Json
        $tileArgs = @('-gUSE_TILE_PROGRAM=1',
            ('-gTILE_PROGRAM_WORDS=' + $tileManifest.issue_count),
            ('-gTILE_DEPENDENCY_WORDS=' + $tileManifest.dependency_word_count),
            ('+TILE_PROGRAM_WORDS=' + $tileManifest.issue_count),
            ('+TILE_DEPENDENCY_WORDS=' + $tileManifest.dependency_word_count),
            ('+TILE_PROGRAM_BIN=' + (Join-Path $RunDirectory 'block13_tile_program.bin').Replace('\','/')),
            ('+TILE_DEPENDENCY_BIN=' + (Join-Path $RunDirectory 'block13_tile_dependencies.bin').Replace('\','/')))
        $topIndex = [Array]::IndexOf($simArgs, 'work.tb_supra2_block_ddr4_axi512_system')
        $simArgs = @($simArgs[0..($topIndex-1)]) + $tileArgs + @($simArgs[$topIndex..($simArgs.Count-1)])
    }
    if ($CalibrationOnly) { $simArgs += '+CALIB_ONLY=1' }
    Write-Output "AXI512_DDR64_RUN_DIRECTORY=$RunDirectory"
    & (Join-Path $QuestaRoot 'vsim.exe') @simArgs
    if ($LASTEXITCODE) { throw "Simulation failed: $RunDirectory\system.log" }
    $log = Get-Content system.log -Raw
    $marker = if ($CalibrationOnly) { 'BOARD_MIG_AXI512_DDR64_CALIB_PASS' }
        else { 'SUPRA2_BLOCK13_DDR_AXI512_PASS' }
    if ($log -notmatch $marker -or $log -match '(?m)^# \*\* (Fatal|Error)') {
        throw "Simulation did not satisfy its required PASS contract: $RunDirectory\system.log"
    }
    Write-Output "AXI512_DDR64_VERIFIED_LOG=$RunDirectory\system.log"
} finally {
    Pop-Location
}
