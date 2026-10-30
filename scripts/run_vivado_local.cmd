@echo off
setlocal
set "PROCESSOR_ARCHITECTURE=AMD64"
set "RDI_BINROOT=D:/soft/vitis/Vivado/2021.1/bin"
set "RDI_PATCHROOT=D:/graduation_design/npu-main/supra2_img_fpga/build/vivado_runtime"
set "PATH=D:/soft/vitis/Vivado/2021.1/bin/unwrapped/win64.o/Microsoft.VC141.CRT;D:/soft/vitis/Vivado/2021.1/bin/unwrapped/win64.o;D:/soft/vitis/Vivado/2021.1/tps/win64/python-3.8.3;D:/soft/vitis/Vivado/2021.1/tps/tcl/tcl8.5/bin;D:/soft/vitis/Vivado/2021.1/tps/mingw/6.2.0/win64.o/nt/bin;%PATH%"
set "RDI_VERBOSE=False"
call "%RDI_BINROOT%/loader.bat" -exec vivado.exe %*
exit /b %ERRORLEVEL%
