param(
    [Parameter(Mandatory=$true)][string]$FileList,
    [Parameter(Mandatory=$true)][string]$Top,
    [Parameter(Mandatory=$true)][string]$Snapshot,
    [Parameter(Mandatory=$true)][string]$PassPattern,
    [string[]]$TestPlusArgs = @(),
    [string[]]$Defines = @(),
    [string[]]$TopGenerics = @(),
    [switch]$StreamOutput,
    [switch]$ReuseSnapshot,
    [ValidateSet('off','typical','all')][string]$DebugLevel = 'typical',
    [string]$VivadoRoot = 'D:\soft\vitis\Vivado\2021.1',
    [string]$Msys2Root = 'D:\C\msys2\ucrt64'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$vivadoBin = Join-Path $VivadoRoot 'bin\unwrapped\win64.o'
$compat = Join-Path $repoRoot 'build\xsim_compat'
$compatData = Join-Path $compat 'data'
$compatXsim = Join-Path $compatData 'xsim'
$compatNt = Join-Path $compat 'tps\mingw\6.2.0\win64.o\nt'
$work = "build/xsim_work_$Snapshot"
$objDir = "xsim.dir/$Snapshot/obj"
$kernel = "xsim.dir/$Snapshot/xsimk.exe"
$gcc = Join-Path $compatNt 'bin\gcc.exe'

foreach ($path in @(
    (Join-Path $vivadoBin 'xvlog.exe'),
    (Join-Path $vivadoBin 'xelab.exe'),
    (Join-Path $Msys2Root 'bin\gcc.exe'),
    (Join-Path $VivadoRoot 'data\xsim\xsim.ini'),
    (Join-Path $VivadoRoot 'lib\win64.o\librdi_simulator_kernel.dll')
)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Required XSim file not found: $path" }
}

if (-not (Test-Path (Join-Path $compatXsim 'xsim.ini'))) {
    New-Item -ItemType Directory -Force -Path $compatXsim | Out-Null
    Copy-Item (Join-Path $VivadoRoot 'data\xsim\xsim.ini') $compatXsim
    foreach ($relative in @('include','vhdl\std','vhdl\std_2008','vhdl\ieee',
                            'vhdl\ieee_2008','system_verilog\uvm')) {
        $source = Join-Path $VivadoRoot "data\xsim\$relative"
        $destination = Join-Path $compatXsim $relative
        New-Item -ItemType Directory -Force -Path (Split-Path $destination -Parent) | Out-Null
        Copy-Item $source $destination -Recurse -Force
    }
}

if (-not (Test-Path $gcc)) {
    New-Item -ItemType Directory -Force -Path $compatNt | Out-Null
    Copy-Item (Join-Path $Msys2Root 'bin'),(Join-Path $Msys2Root 'include'),
        (Join-Path $Msys2Root 'lib') -Destination $compatNt -Recurse -Force
}
if (-not (Test-Path $gcc)) { throw "Compatible GCC was not staged at $gcc" }

Push-Location $repoRoot
$oldPath = $env:PATH
$oldData = $env:RDI_DATADIR
$oldApp = $env:RDI_APPROOT
try {
    $env:PATH = "$(Join-Path $VivadoRoot 'lib\win64.o');$(Join-Path $vivadoBin 'Microsoft.VC141.CRT');$(Join-Path $compatNt 'bin');$(Join-Path $compatNt 'lib\gcc\x86_64-w64-mingw32\14.2.0');$vivadoBin;$env:PATH"
    $env:RDI_DATADIR = $compatData
    $env:RDI_APPROOT = $VivadoRoot
    $env:RDI_BINROOT = Join-Path $VivadoRoot 'bin'
    $env:RDI_LIBDIR = Join-Path $VivadoRoot 'lib\win64.o'
    $env:RDI_MINGW_LIB = "$(Join-Path $compatNt 'bin');$(Join-Path $compatNt 'lib\gcc\x86_64-w64-mingw32\14.2.0')"
    if ($ReuseSnapshot) {
        if (-not (Test-Path -LiteralPath $kernel)) {
            throw "Cannot reuse missing XSim snapshot kernel: $kernel"
        }
    }
    else {
        New-Item -ItemType Directory -Force -Path $work | Out-Null

        $xvlogArgs = @('-sv', '--work', "work=$work")
        foreach ($define in $Defines) { $xvlogArgs += @('-d', $define) }
        $xvlogArgs += @('-f', $FileList)
        $xvlogLog = & (Join-Path $vivadoBin 'xvlog.exe') @xvlogArgs 2>&1
        $xvlogLog | Tee-Object -FilePath "build/xsim_$Snapshot.xvlog.log"
        if ($LASTEXITCODE -ne 0) { throw "xvlog failed; see build/xsim_$Snapshot.xvlog.log" }

        $xelabArgs = @('-a', '--initfile', (Join-Path $compatXsim 'xsim.ini'),
            '-debug', $DebugLevel, '-L', "work=$work", '-s', $Snapshot)
        foreach ($generic in $TopGenerics) {
            $xelabArgs += @('-generic_top', $generic)
        }
        $xelabArgs += "work.$Top"
        # xelab writes expected linker-failure diagnostics to stderr; under
        # $ErrorActionPreference='Stop' PowerShell 5.1 surfaces those as
        # NativeCommandError, so relax EAP while invoking the native tool.
        $savedEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $xelabLog = & (Join-Path $vivadoBin 'xelab.exe') @xelabArgs 2>&1
            $xelabExit = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedEap
        }
        # xelab's own final link is expected to fail with ADDR32 relocation
        # truncation; the script relinks the emitted objects below with an
        # explicit low image base, so the exit code is ignored by design.
        $xelabLog | Set-Content -Path "build/xsim_$Snapshot.xelab.log"
        $objects = @(Get-ChildItem -LiteralPath $objDir -Filter 'xsim_*.win64.obj' |
            Sort-Object { [int]($_.BaseName -replace '^xsim_(\d+)\.win64$', '$1') })
        if ($objects.Count -lt 2 -or
            -not ($objects | Where-Object { $_.Name -eq 'xsim_0.win64.obj' }) -or
            -not ($objects | Where-Object { $_.Name -eq 'xsim_1.win64.obj' })) {
            throw "xelab emitted an incomplete standalone object set under $objDir"
        }

        $linkArgs = @(
            '-Wa,-W','-O','-Wl,--stack,104857600','-Wl,--image-base,0x10000000',
            '-o',$kernel
        )
        $linkArgs += @($objects | ForEach-Object { $_.FullName })
        $linkArgs += @(
            "-L$(Join-Path $VivadoRoot 'lib\win64.o')",
            '-lrdi_simulator_kernel','-lrdi_simbridge_kernel'
        )
        & $gcc @linkArgs
        if ($LASTEXITCODE -ne 0) { throw 'Standalone XSim kernel link failed' }
        Write-Host 'XSim standalone kernel linked with the local compatibility toolchain.'
    }

    $runtimeArgs = @('--runall')
    foreach ($argument in $TestPlusArgs) {
        $runtimeArgs += @('--testplusarg', $argument)
    }
    if ($StreamOutput) {
        $savedEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $simLog = & $kernel @runtimeArgs 2>&1 |
                Tee-Object -FilePath "build/xsim_$Snapshot.sim.log"
            $simExit = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $savedEap }
    }
    else {
        $simLog = & $kernel @runtimeArgs 2>&1
        $simExit = $LASTEXITCODE
        $simLog | Tee-Object -FilePath "build/xsim_$Snapshot.sim.log"
    }
    $simText = $simLog -join "`n"
    if ($simExit -ne 0 -or $simText -notmatch $PassPattern -or
        $simText -match 'Fatal:|ERROR:|FATAL:') {
        throw "Standalone XSim test failed; see build/xsim_$Snapshot.sim.log"
    }
}
finally {
    $env:PATH = $oldPath
    $env:RDI_DATADIR = $oldData
    $env:RDI_APPROOT = $oldApp
    Pop-Location
}
