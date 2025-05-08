module @uniform_rng {
  func.func private @NextOffsetFunc() -> tensor<i64> attributes {byre_compute_name = "NextOffset", byre_force_compute_name}
  func.func private @GetSeedFunc() -> tensor<i64> attributes {byre_compute_name = "GetSeed", byre_force_compute_name}
  func.func private @Unknown0(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<2x128x128xf32> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f32>
    %2 = mhlo.custom_call @byteir.rng_uniform(%0, %1, %arg0, %arg1) {backend_config = ""} : (tensor<f32>, tensor<f32>, tensor<i64>, tensor<i64>) -> tensor<2x128x128xf32>
    %3 = mhlo.add %2, %2 : tensor<2x128x128xf32>
    return %3 : tensor<2x128x128xf32>
  }
  func.func @uniform_rngf32() -> tensor<2x128x128xf32> {
    %0 = call @GetSeedFunc() : () -> tensor<i64>
    %1 = call @NextOffsetFunc() : () -> tensor<i64>
    %2 = call @Unknown0(%0, %1) : (tensor<i64>, tensor<i64>) -> tensor<2x128x128xf32>
    return %2 : tensor<2x128x128xf32>
  }
  func.func private @Unknown1(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<2x128x128xf64> attributes {__byteir_elementwise_fusion__} {
    %0 = mhlo.constant dense<0.000000e+00> : tensor<f64>
    %1 = mhlo.constant dense<1.000000e+00> : tensor<f64>
    %2 = mhlo.custom_call @byteir.rng_uniform(%0, %1, %arg0, %arg1) {backend_config = ""} : (tensor<f64>, tensor<f64>, tensor<i64>, tensor<i64>) -> tensor<2x128x128xf64>
    %3 = mhlo.add %2, %2 : tensor<2x128x128xf64>
    return %3 : tensor<2x128x128xf64>
  }
  func.func @uniform_rngf64() -> tensor<2x128x128xf64> {
    %0 = call @GetSeedFunc() : () -> tensor<i64>
    %1 = call @NextOffsetFunc() : () -> tensor<i64>
    %2 = call @Unknown1(%0, %1) : (tensor<i64>, tensor<i64>) -> tensor<2x128x128xf64>
    return %2 : tensor<2x128x128xf64>
  }
}

