#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<(d0) -> ((d0 floordiv 32) * 64)>
module {
  func.func private @Unknown0(%arg0: memref<5376x2048xf16>, %arg1: memref<2048x5376xf16>) -> memref<5376x5376xf16> attributes {__byteir_gemm_block_size__ = [64, 2, 1], __byteir_gemm_pipeline_depth__ = 3 : i64, __byteir_gemm_tile_config__ = [128, 128, 32], __byteir_matmul_epilogue_fusion__} {
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<5376x5376xf16>
    scf.forall (%arg2, %arg3) in (42, 42) {
      %0 = affine.apply #map(%arg2)
      %1 = affine.apply #map(%arg3)
      %subview = memref.subview %alloc[%0, %1] [128, 128] [1, 1] : memref<5376x5376xf16> to memref<128x128xf16, strided<[5376, 1], offset: ?>>
      %thread_id_x = gpu.thread_id  x
      %thread_id_y = gpu.thread_id  y
      %2 = affine.apply #map1()[%thread_id_y]
      scf.for %arg4 = %2 to %c128 step %c128 {
        %3 = affine.apply #map2(%thread_id_x)
        scf.for %arg5 = %3 to %c128 step %c128 {
          %subview_0 = memref.subview %subview[%arg4, %arg5] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
          linalg.fill {__internal_linalg_transform__ = "vectorize"} ins(%cst : f16) outs(%subview_0 : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
        }
      }
      scf.for %arg4 = %c0 to %c2048 step %c32 {
        %subview_0 = memref.subview %arg0[%0, %arg4] [128, 32] [1, 1] : memref<5376x2048xf16> to memref<128x32xf16, strided<[2048, 1], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg4, %1] [32, 128] [1, 1] : memref<2048x5376xf16> to memref<32x128xf16, strided<[5376, 1], offset: ?>>
        scf.for %arg5 = %2 to %c128 step %c128 {
          %3 = affine.apply #map2(%thread_id_x)
          scf.for %arg6 = %3 to %c128 step %c128 {
            %subview_2 = memref.subview %subview_0[%arg5, 0] [64, 32] [1, 1] : memref<128x32xf16, strided<[2048, 1], offset: ?>> to memref<64x32xf16, strided<[2048, 1], offset: ?>>
            %subview_3 = memref.subview %subview_1[0, %arg6] [32, 64] [1, 1] : memref<32x128xf16, strided<[5376, 1], offset: ?>> to memref<32x64xf16, strided<[5376, 1], offset: ?>>
            %subview_4 = memref.subview %subview[%arg5, %arg6] [64, 64] [1, 1] : memref<128x128xf16, strided<[5376, 1], offset: ?>> to memref<64x64xf16, strided<[5376, 1], offset: ?>>
            linalg.matmul {__byteir_gpu_tile_gemm_0, __byteir_mma__, __byteir_mma_level__ = "Threadblock", __byteir_target__ = "nv_sm_80", __internal_linalg_transform__ = "vectorize"} ins(%subview_2, %subview_3 : memref<64x32xf16, strided<[2048, 1], offset: ?>>, memref<32x64xf16, strided<[5376, 1], offset: ?>>) outs(%subview_4 : memref<64x64xf16, strided<[5376, 1], offset: ?>>)
          }
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %alloc : memref<5376x5376xf16>
  }
}

