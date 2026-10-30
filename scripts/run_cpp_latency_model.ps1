param(
    [string]$Compiler = 'D:\C\msys2\ucrt64\bin\g++.exe',
    [string]$BuildDirectory = 'build\cpp_latency_msys2'
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$compilerPath = (Resolve-Path -LiteralPath $Compiler).Path
$compilerBin = Split-Path -Parent $compilerPath
$make = Join-Path $compilerBin 'mingw32-make.exe'
if (!(Test-Path -LiteralPath $make)) { throw "Missing make tool: $make" }
$build = if ([IO.Path]::IsPathRooted($BuildDirectory)) {
    $BuildDirectory
} else { Join-Path $repo $BuildDirectory }
$previousPath = $env:PATH
try {
    $env:PATH = "$compilerBin;$previousPath"
    Push-Location $repo
    try {
        & cmake -S simulator/cpp_latency -B $build -G 'MinGW Makefiles' `
            "-DCMAKE_CXX_COMPILER=$compilerPath" "-DCMAKE_MAKE_PROGRAM=$make" `
            -DCMAKE_BUILD_TYPE=Debug
        if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
        & cmake --build $build -j 4
        if ($LASTEXITCODE -ne 0) { throw 'C++ build failed' }
        & ctest --test-dir $build --output-on-failure |
            Tee-Object -FilePath (Join-Path $build 'verification.log')
        if ($LASTEXITCODE -ne 0) { throw 'C++ regression failed' }
        & (Join-Path $build 'npu_pipeline_demo.exe') |
            Tee-Object -FilePath (Join-Path $build 'pipeline_demo.log')
        if ($LASTEXITCODE -ne 0) { throw 'Pipeline example failed' }
        & (Join-Path $build 'mig_axi_model.exe') --requests simulator/cpp_latency/examples/mig_axi_requests.csv |
            Tee-Object -FilePath (Join-Path $build 'mig_example.log')
        if ($LASTEXITCODE -ne 0) { throw 'MIG example failed' }
        & (Join-Path $build 'ddr4_replay.exe') --trace simulator/cpp_latency/examples/ddr4_pin_trace_example.csv |
            Tee-Object -FilePath (Join-Path $build 'ddr_example.log')
        if ($LASTEXITCODE -ne 0) { throw 'DDR example replay failed' }
        & (Join-Path $build 'supra2_latency_model.exe') `
            --workload simulator/cpp_latency/examples/block13_reference.json `
            --config simulator/cpp_latency/examples/model_config.json `
            --output (Join-Path $build 'block13_result.json')
        if ($LASTEXITCODE -ne 0) { throw 'Measured stage ledger failed' }
        & (Join-Path $build 'model_characterization.exe') (Join-Path $build 'characterization.json')
        if ($LASTEXITCODE -ne 0) { throw 'Traffic characterization failed' }
        & (Join-Path $build 'attention_overlap_model.exe') `
            --report (Join-Path $build 'attention_overlap_result.json') `
            --trace (Join-Path $build 'attention_overlap_group_trace.csv')
        if ($LASTEXITCODE -ne 0) { throw 'Attention overlap estimation failed' }
        & (Join-Path $build 'dma_width_model.exe') (Join-Path $build 'dma_width_result.csv')
        if ($LASTEXITCODE -ne 0) { throw 'DMA width estimation failed' }
        & (Join-Path $build 'gemm_architecture_model.exe') (Join-Path $build 'gemm_architecture_result.csv')
        if ($LASTEXITCODE -ne 0) { throw 'GEMM architecture estimation failed' }
    } finally { Pop-Location }
} finally { $env:PATH = $previousPath }
