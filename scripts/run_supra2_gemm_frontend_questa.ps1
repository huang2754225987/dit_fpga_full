$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$questa='D:\soft\questa\win64'
$env:LM_LICENSE_FILE='C:\ProgramData\MentorGraphics\License\license.dat'
$env:MGLS_LICENSE_FILE=$env:LM_LICENSE_FILE
$runTag=Get-Date -Format 'yyyyMMdd_HHmmss'
$workDir=Join-Path $repo "build\supra2_gemm_frontend\questa_runs\$runTag"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null
$cases=@(
    @{ Name='x_embed'; K=16; H=16; N=576; Dir='supra2_x_embed_qlinear' },
    @{ Name='qkv'; K=576; H=64; N=1728; Dir='supra2_qkv_projection_fp32_frontend' },
    @{ Name='mlp2'; K=2304; H=256; N=576; Dir='supra2_mlp2_projection_fp32_frontend' }
)
foreach($case in $cases){
    $fixture=Join-Path $repo "build\$($case.Dir)"
    foreach($item in @(
        @{ Source='raw_input_fp16.hex'; Dest='raw'; Count=32*$case.K },
        @{ Source='weight_int8.hex'; Dest='weight'; Count=32*$case.K },
        @{ Source='weight_scale_fp32.hex'; Dest='scale'; Count=32 },
        @{ Source='bias_fp32.hex'; Dest='bias'; Count=32 }
    )){
        $words=@(Get-Content (Join-Path $fixture $item.Source) -TotalCount $item.Count)
        if($words.Count -ne $item.Count){throw 'Incomplete GEMM fixture'}
        $words | Set-Content (Join-Path $workDir "$($case.Name)_$($item.Dest).hex")
    }
    $reference=@(Get-Content (Join-Path $fixture 'reference_output_fp16.hex') -TotalCount (32*$case.N))
    if($reference.Count -ne 32*$case.N){throw 'Incomplete GEMM expected outputs'}
    $expected=for($row=0;$row -lt 32;$row++){ $reference[($row*$case.N)..($row*$case.N+31)] }
    $expected | Set-Content (Join-Path $workDir "$($case.Name)_expected.hex")
}
Push-Location $workDir
try{
    & (Join-Path $questa 'vlib.exe') work
    if($LASTEXITCODE -ne 0){throw 'vlib failed'}
    $sources=@('rtl/pkg/fp16_utils_pkg.sv','rtl/gemm/mac_int8.sv','rtl/gemm/pe.sv',
        'rtl/gemm/systolic_array.sv','rtl/supra2/fp32_add_rne_pipeline.sv',
        'rtl/supra2/hadamard_transform.sv',
        'rtl/supra2/activation_quantizer.sv','rtl/supra2/qlinear_epilogue.sv',
        'rtl/supra2/supra2_sync_word_ram.sv',
        'rtl/supra2/supra2_gemm_engine.sv','rtl/sim/tb_supra2_gemm_frontend.sv') |
        ForEach-Object {Join-Path $repo $_}
    & (Join-Path $questa 'vlog.exe') -sv -work work @sources
    if($LASTEXITCODE -ne 0){throw 'vlog failed'}
    foreach($case in $cases){
        $sim=& (Join-Path $questa 'vsim.exe') -c -lib work tb_supra2_gemm_frontend `
            "+K=$($case.K)" "+H=$($case.H)" "+RAW=$($case.Name)_raw.hex" `
            "+WEIGHT=$($case.Name)_weight.hex" "+SCALE=$($case.Name)_scale.hex" `
            "+BIAS=$($case.Name)_bias.hex" "+EXPECTED=$($case.Name)_expected.hex" `
            -do 'run -all; quit -f' 2>&1
        $simExit=$LASTEXITCODE
        $sim | Set-Content "$($case.Name)_questa.log"
        $sim
        $simText=$sim -join "`n"
        if($simExit -ne 0 -or $simText -notmatch "SUPRA2_GEMM_FRONTEND_PASS M=32 N=32 K=$($case.K) H=$($case.H)" -or
            $simText -match '\*\* (Fatal|Error)|Errors: [1-9]'){
            throw "GEMM frontend failed for $($case.Name)"
        }
    }
    "Questa integrated GEMM frontend golden regression PASS; logs: $workDir"
}finally{Pop-Location}
