BYTEIR_BIN=../../../compiler/build/bin/

# FileCheck测试通过
${BYTEIR_BIN}/byteir-opt -gpu-distributed-to-warp  -canonicalize -cse --verify-diagnostics ./gpu-distributed-to-warp.mlir | FileCheck ./gpu-distributed-to-warp.mlir
${BYTEIR_BIN}/byteir-opt -gpu-distributed-to-warp  -canonicalize -cse --verify-diagnostics ./gpu-distributed-to-warp.mlir -o gpu-distributed.mlir 
