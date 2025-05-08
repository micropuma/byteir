cmake -B./compiler/build \
      -H./compiler/cmake \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DPython3_EXECUTABLE=/mnt/home/douliyang/mlir-workspace/byteir/venv/bin/python \
      -DLLVM_INSTALL_PATH=$(pwd)/external/llvm-project/build/install \
      -DLLVM_EXTERNAL_LIT=/mnt/home/douliyang/mlir-workspace/byteir/venv/bin/lit \
      -Dpybind11_DIR=/mnt/home/douliyang/mlir-workspace/byteir/venv/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 \
      -Dnanobind_DIR=/mnt/home/douliyang/mlir-workspace/byteir/venv/lib/python3.11/site-packages/nanobind/cmake \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -DBYTEIR_ENABLE_BINDINGS_PYTHON=ON

cmake --build ./compiler/build --target check-byteir
cmake --build ./compiler/build --target byteir-python-pack