cmake -H./runtime/cmake \
      -B./runtime/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_INSTALL_PATH=$(pwd)/external/llvm-project/build/install \
      -DCMAKE_INSTALL_PREFIX="$(pwd)/runtime/build/install" \
      -Dbrt_ENABLE_PYTHON_BINDINGS=ON \
      -Dbrt_USE_CUDA=ON

cmake --build ./runtime/build --target all --target install
