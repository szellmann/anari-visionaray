// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

// std
#include <cstdint>
#ifdef __HIPCC__
// hip
#include <hip/hip_runtime.h>
#endif
// visionaray
#include "visionaray/detail/parallel_for.h"
#include "visionaray/detail/thread_pool.h"
#include "visionaray/math/math.h"

namespace visionaray {
  namespace serial {
    template <typename Func>
    void for_each(int32_t xmin, int32_t xmax, Func func)
    {
      for (int32_t x = xmin; x != xmax; ++x) {
        func(x);
      }
    }

    template <typename Func>
    void for_each(int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  Func func)
    {
      for (int32_t y = ymin; y != ymax; ++y) {
        for (int32_t x = xmin; x != xmax; ++x) {
          func(x, y);
        }
      }
    }

    template <typename Func>
    void for_each(int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  int32_t zmin, int32_t zmax,
                  Func func)
    {
      for (int32_t z = zmin; z != zmax; ++z) {
        for (int32_t y = ymin; y != ymax; ++y) {
          for (int32_t x = xmin; x != xmax; ++x) {
            func(x, y, z);
          }
        }
      }
    }
  } // serial

  namespace parallel {
    template <typename Func>
    void for_each(thread_pool &pool,
                  int32_t xmin, int32_t xmax,
                  Func func)
    {
      parallel_for(pool,
          tiled_range1d<int32_t>(xmin, xmax, 1024),
          [&](range1d<int32_t> r) {
              for (int x = r.begin(); x != r.end(); ++x) {
                func(x);
              }
          });
    }

    template <typename Func>
    void for_each(thread_pool &pool,
                  int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  Func func)
    {
      parallel_for(pool,
          tiled_range2d<int32_t>(xmin, xmax, 64, ymin, ymax, 64),
          [&](range2d<int32_t> r) {
            for (int y = r.cols().begin(); y != r.cols().end(); ++y) {
              for (int x = r.rows().begin(); x != r.rows().end(); ++x) {
                func(x, y);
              }
            }
          });
    }

  } // parallel

#ifdef _OPENMP
  namespace omp {
    template <typename Func>
    void for_each(int32_t xmin, int32_t xmax, Func func)
    {
      #pragma omp parallel for
      for (int32_t x = xmin; x != xmax; ++x) {
        func(x);
      }
    }

    template <typename Func>
    void for_each(int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  int32_t zmin, int32_t zmax,
                  Func func)
    {
      #pragma omp parallel for collapse(3)
      for (int32_t z = zmin; z != zmax; ++z) {
        for (int32_t y = ymin; y != ymax; ++y) {
          for (int32_t x = xmin; x != xmax; ++x) {
            func(x, y, z);
          }
        }
      }
    }
  }
#endif

#ifdef __CUDACC__
  namespace cuda {
    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax, Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;

      if (x < xmin || x >= xmax)
        return;

      func(x);
    }

    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                    int32_t ymin, int32_t ymax,
                                    Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < xmin || x >= xmax || y < ymin || y >= ymax)
        return;

      func(x, y);
    }

    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                    int32_t ymin, int32_t ymax,
                                    int32_t zmin, int32_t zmax,
                                    Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
      int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

      if (x < xmin || x >= xmax || y < ymin || y >= ymax || z < zmin || z >= zmax)
        return;

      func(x, y, z);
    }

    template <typename Func>
    void for_each(cudaStream_t stream,
                  int32_t xmin, int32_t xmax,
                  Func func)
    {
      dim3 blockSize = 256;
      dim3 gridSize = div_up(xmax-xmin, (int)blockSize.x);

      for_each_kernel<<<gridSize, blockSize, 0, stream>>>(xmin, xmax, func);
    }

    template <typename Func>
    void for_each(cudaStream_t stream,
                  int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  Func func)
    {
      dim3 blockSize = 64;
      dim3 gridSize(
              div_up(xmax-xmin, (int)blockSize.x),
              div_up(ymax-ymin, (int)blockSize.y)
              );

      for_each_kernel<<<gridSize, blockSize, 0, stream>>>(xmin, xmax,
                                                          ymin, ymax,
                                                          func);
    }

    template <typename Func>
    void for_each(cudaStream_t stream,
                  int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  int32_t zmin, int32_t zmax,
                  Func func)
    {
      dim3 blockSize(8, 8, 8);
      dim3 gridSize(
              div_up(xmax-xmin, (int)blockSize.x),
              div_up(ymax-ymin, (int)blockSize.y),
              div_up(zmax-zmin, (int)blockSize.z)
              );

      for_each_kernel<<<gridSize, blockSize, 0, stream>>>(xmin, xmax,
                                                          ymin, ymax,
                                                          zmin, zmax,
                                                          func);
    }
  } // cuda
#endif

#ifdef __HIPCC__
  namespace hip {
    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax, Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;

      if (x < xmin || x >= xmax)
        return;

      func(x);
    }

    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                    int32_t ymin, int32_t ymax,
                                    Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

      if (x < xmin || x >= xmax || y < ymin || y >= ymax)
        return;

      func(x, y);
    }

    template <typename Func>
    __global__ void for_each_kernel(int32_t xmin, int32_t xmax,
                                    int32_t ymin, int32_t ymax,
                                    int32_t zmin, int32_t zmax,
                                    Func func)
    {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
      int32_t z = blockIdx.z * blockDim.z + threadIdx.z;

      if (x < xmin || x >= xmax || y < ymin || y >= ymax || z < zmin || z >= zmax)
        return;

      func(x, y, z);
    }

    template <typename Func>
    void for_each(hipStream_t stream,
                  int32_t xmin, int32_t xmax,
                  Func func)
    {
      dim3 blockSize = 256;
      dim3 gridSize = div_up(xmax-xmin, (int)blockSize.x);

      hipLaunchKernelGGL(
        for_each_kernel<Func>, gridSize, blockSize, 0, stream, xmin, xmax, func);
    }

    template <typename Func>
    void for_each(hipStream_t stream,
                  int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  Func func)
    {
      dim3 blockSize = 64;
      dim3 gridSize(
              div_up(xmax-xmin, (int)blockSize.x),
              div_up(ymax-ymin, (int)blockSize.y)
              );

      hipLaunchKernelGGL(
        for_each_kernel<Func>, gridSize, blockSize, 0, stream,
        xmin, xmax, ymin, ymax, func);
    }

    template <typename Func>
    void for_each(hipStream_t stream,
                  int32_t xmin, int32_t xmax,
                  int32_t ymin, int32_t ymax,
                  int32_t zmin, int32_t zmax,
                  Func func)
    {
      dim3 blockSize(8, 8, 8);
      dim3 gridSize(
              div_up(xmax-xmin, (int)blockSize.x),
              div_up(ymax-ymin, (int)blockSize.y),
              div_up(zmax-zmin, (int)blockSize.z)
              );

      hipLaunchKernelGGL(
        for_each_kernel<Func>, gridSize, blockSize, 0, stream,
        xmin, xmax, ymin, ymax, zmin, zmax, func);
    }
  } // hip
#endif

} // visionaray
