param([string]$QuestaRoot='D:\soft\questa\win64')
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$license='C:\ProgramData\MentorGraphics\License\license.dat'
foreach($file in @((Join-Path $QuestaRoot 'vlib.exe'),(Join-Path $QuestaRoot 'vlog.exe'),
                   (Join-Path $QuestaRoot 'vsim.exe'),$license)) {
    if(!(Test-Path -LiteralPath $file)){throw "Required Questa input missing: $file"}
}
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'
$run=Join-Path $repo "build\supra2_gemm_overlap\questa_runs\$tag"
New-Item -ItemType Directory -Force -Path $run | Out-Null
$env:LM_LICENSE_FILE=$license
$env:MGLS_LICENSE_FILE=$license
$env:PATH="$QuestaRoot;$env:PATH"
$sources=@('rtl/pkg/fp16_utils_pkg.sv','rtl/gemm/mac_int8.sv','rtl/gemm/pe.sv',
    'rtl/gemm/systolic_array.sv','rtl/supra2/fp32_add_rne_pipeline.sv',
    'rtl/supra2/hadamard_transform.sv','rtl/supra2/activation_quantizer.sv',
    'rtl/supra2/qlinear_epilogue.sv','rtl/supra2/supra2_sync_word_ram.sv',
    'rtl/supra2/supra2_gemm_engine.sv','rtl/sim/tb_supra2_gemm_frontend_overlap.sv',
    'rtl/sim/tb_supra2_gemm_drain_overlap.sv','rtl/sim/tb_supra2_gemm_weight_pingpong.sv',
    'rtl/sim/tb_supra2_gemm_frontend_pingpong_golden.sv') |
    ForEach-Object {Join-Path $repo $_}
$cases=@(
    @{Top='tb_supra2_gemm_frontend_overlap'; Pass='SUPRA2_GEMM_FRONTEND_OVERLAP_PASS'},
    @{Top='tb_supra2_gemm_drain_overlap'; Pass='SUPRA2_GEMM_DRAIN_OVERLAP_PASS'},
    @{Top='tb_supra2_gemm_weight_pingpong'; Pass='SUPRA2_GEMM_WEIGHT_PINGPONG_PASS'}
)
Push-Location $run
try {
    & (Join-Path $QuestaRoot 'vlib.exe') work
    if($LASTEXITCODE -ne 0){throw 'vlib failed'}
    & (Join-Path $QuestaRoot 'vlog.exe') -sv -work work -l compile.log @sources
    if($LASTEXITCODE -ne 0){throw 'vlog failed'}
    foreach($case in $cases) {
        $sim=& (Join-Path $QuestaRoot 'vsim.exe') -c -lib work $case.Top `
            -do 'run -all; quit -f' 2>&1
        $status=$LASTEXITCODE
        $sim | Set-Content "$($case.Top).log"
        $text=$sim -join "`n"
        if($status -ne 0 -or $text -notmatch $case.Pass -or
           $text -match '\*\* (Fatal|Error)|Errors: [1-9]') {
            throw "Questa $($case.Top) failed; see $run"
        }
        ($sim | Select-String -Pattern $case.Pass) | Write-Output
    }
    $fixture=Join-Path $repo 'build\supra2_qkv_projection_fp32_frontend'
    foreach($name in @('raw_input_fp16.hex','weight_int8.hex','weight_scale_fp32.hex',
                        'bias_fp32.hex','reference_output_fp16.hex')) {
        if(!(Test-Path -LiteralPath (Join-Path $fixture $name))) {throw "Missing QKV fixture $name"}
    }
    $reader=[IO.StreamReader]::new((Join-Path $fixture 'reference_output_fp16.hex'))
    $writer=[IO.StreamWriter]::new((Join-Path $run 'pingpong_expected.hex'))
    $writerWide=[IO.StreamWriter]::new((Join-Path $run 'drain_expected.hex'))
    try {
        for($row=0;$row -lt 96;$row++) {
            for($col=0;$col -lt 1728;$col++) {
                $line=$reader.ReadLine()
                if($null -eq $line){throw 'Incomplete QKV golden tensor'}
                if($col -lt 32){$writer.WriteLine($line)}
                if($row -lt 32 -and $col -lt 64){$writerWide.WriteLine($line)}
            }
        }
    } finally { $reader.Dispose(); $writer.Dispose(); $writerWide.Dispose() }
    $sim=& (Join-Path $QuestaRoot 'vsim.exe') -c -lib work tb_supra2_gemm_frontend_pingpong_golden `
        "+RAW=$fixture/raw_input_fp16.hex" "+WEIGHT=$fixture/weight_int8.hex" `
        "+SCALE=$fixture/weight_scale_fp32.hex" "+BIAS=$fixture/bias_fp32.hex" `
        '+EXPECTED=pingpong_expected.hex' -do 'run -all; quit -f' 2>&1
    $status=$LASTEXITCODE
    $sim | Set-Content 'tb_supra2_gemm_frontend_pingpong_golden.log'
    $text=$sim -join "`n"
    if($status -ne 0 -or $text -notmatch 'SUPRA2_GEMM_PINGPONG_GOLDEN_PASS' -or
       $text -match '\*\* (Fatal|Error)|Errors: [1-9]') {throw "Golden ping-pong failed; see $run"}
    ($sim | Select-String -Pattern 'SUPRA2_GEMM_PINGPONG_GOLDEN_PASS') | Write-Output
    $sim=& (Join-Path $QuestaRoot 'vsim.exe') -c -lib work -gM=32 -gN=64 tb_supra2_gemm_frontend_pingpong_golden `
        "+RAW=$fixture/raw_input_fp16.hex" "+WEIGHT=$fixture/weight_int8.hex" `
        "+SCALE=$fixture/weight_scale_fp32.hex" "+BIAS=$fixture/bias_fp32.hex" `
        '+EXPECTED=drain_expected.hex' -do 'run -all; quit -f' 2>&1
    $status=$LASTEXITCODE
    $sim | Set-Content 'tb_supra2_gemm_drain_golden.log'
    $text=$sim -join "`n"
    if($status -ne 0 -or $text -notmatch 'SUPRA2_GEMM_PINGPONG_GOLDEN_PASS M=32 N=64' -or
       $text -match '\*\* (Fatal|Error)|Errors: [1-9]') {throw "Golden drain failed; see $run"}
    ($sim | Select-String -Pattern 'SUPRA2_GEMM_PINGPONG_GOLDEN_PASS') | Write-Output
    Write-Output "GEMM_OVERLAP_QUESTA_PASS run_directory=$run"
} finally { Pop-Location }
