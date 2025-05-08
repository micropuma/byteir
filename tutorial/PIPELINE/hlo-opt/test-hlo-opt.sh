BYTEIR_BIN=../../../compiler/build/bin/

# FileCheck测试通过
${BYTEIR_BIN}/byteir-opt -hlo-graph-opt -hlo-fusion-opt ./rng.mlir | FileCheck ./rng.mlir
${BYTEIR_BIN}/byteir-opt -hlo-graph-opt -hlo-fusion-opt ./rng.mlir -o rng2.mlir
