BYTEIR="/home/douliyang/large/mlir-workspace/byteir"

export PYTHONPATH="${BYTEIR}/compiler/build/python_packages/byteir"

python3 -m byteir.tools.compiler -v \
  "${BYTEIR}/compiler/test/E2E/CUDA/MLPInference/input.mlir" \
  -o out.mlir \
  --entry_func forward \
  2>&1 | tee pipeline.log
