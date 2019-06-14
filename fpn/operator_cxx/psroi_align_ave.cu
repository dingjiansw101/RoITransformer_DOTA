/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling.cu
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi, Jifeng Dai
 * modified by Jian Ding
*/
#include "./psroi_align_ave-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define PSROIALIGNAVEPOOLING_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
namespace cuda {

  template <typename T>
  __device__ T bilinear_interpolate(
      const T* bottom_data,
      const int height,
      const int width,
      T y,
      T x,
      const int index /* index for debug only*/) {
    // deal with cases that inverse elements are out of feature map boundary
    if (y < -1.0 || y > height || x < -1.0 || x > width) {
      // empty
      return 0;
    }
  
    if (y <= 0) {
      y = 0;
    }
    if (x <= 0) {
      x = 0;
    }
  
    int y_low = static_cast<int>(y);
    int x_low = static_cast<int>(x);
    int y_high;
    int x_high;
  
    if (y_low >= height - 1) {
      y_high = y_low = height - 1;
      y = (T)y_low;
    } else {
      y_high = y_low + 1;
    }
  
    if (x_low >= width - 1) {
      x_high = x_low = width - 1;
      x = (T)x_low;
    } else {
      x_high = x_low + 1;
    }
  
    T ly = y - y_low;
    T lx = x - x_low;
    T hy = 1. - ly, hx = 1. - lx;
    // do bilinear interpolation
    T v1 = bottom_data[y_low * width + x_low];
    T v2 = bottom_data[y_low * width + x_high];
    T v3 = bottom_data[y_high * width + x_low];
    T v4 = bottom_data[y_high * width + x_high];
    T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
  
    T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  
    return val;
  }
  

template <typename DType>
__global__ void PSROIALIGNAVEPoolForwardKernel(
  const int count,
  const DType* bottom_data,
  const DType spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const int sampling_ratio,
  const DType* bottom_rois,
  const int output_dim,
  const int group_size,
  DType* top_data) {
  CUDA_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const DType* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    // DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
    // DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
    // DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale;
    // DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale;
    DType roi_start_w = (offset_bottom_rois[1]) * spatial_scale;
    DType roi_start_h = (offset_bottom_rois[2]) * spatial_scale;
    DType roi_end_w = (offset_bottom_rois[3]) * spatial_scale;
    DType roi_end_h = (offset_bottom_rois[4]) * spatial_scale;

    // Force too small ROIs to be 1x1
    DType roi_width = max(roi_end_w - roi_start_w, (DType)1.);  // avoid 0
    DType roi_height = max(roi_end_h - roi_start_h, (DType)1.);

    // Compute w and h at bottom
    DType bin_size_h = static_cast<DType>(roi_height) / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width) / static_cast<DType>(pooled_width);

    // int hstart = floor(static_cast<DType>(ph) * bin_size_h
    //                     + roi_start_h);
    // int wstart = floor(static_cast<DType>(pw)* bin_size_w
    //                     + roi_start_w);
    // int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h
    //                   + roi_start_h);
    // int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w
    //                   + roi_start_w);
    // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart, 0), height);
    // hend = min(max(hend, 0), height);
    // wstart = min(max(wstart, 0), width);
    // wend = min(max(wend, 0), width);
    // bool is_empty = (hend <= hstart) || (wend <= wstart);



    int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;

    const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
   
       // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const DType sample_count = roi_bin_grid_h * roi_bin_grid_w; // e.g., iy = 0, 1
    DType output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const DType y = roi_start_h + ph * bin_size_h +
          static_cast<DType>(iy + .5f) * bin_size_h /
              static_cast<DType>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const DType x = roi_start_w + pw * bin_size_w +
            static_cast<DType>(ix + .5f) * bin_size_w /
                static_cast<DType>(roi_bin_grid_w);

        DType val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= sample_count;

    top_data[index] = output_val;
    // DType out_sum = 0;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = h*width + w;
    //     out_sum += offset_bottom_data[bottom_index];
    //   }
    // }

    // DType bin_area = (hend - hstart)*(wend - wstart);
    // top_data[index] = is_empty? (DType)0. : out_sum/bin_area;
  }
}

template<typename DType>
inline void PSROIALIGNAVEPoolForward(const Tensor<gpu, 4, DType> &out,
                           const Tensor<gpu, 4, DType> &data,
                           const Tensor<gpu, 2, DType> &bbox,
                           const float spatial_scale,
                           const int sampling_ratio,
                           const int output_dim_,
                           const int group_size_) {
  const DType *bottom_data = data.dptr_;
  const DType *bottom_rois = bbox.dptr_;
  DType *top_data = out.dptr_;
  const int count = out.shape_.Size();
  const int channels = data.size(1);
  const int height = data.size(2);
  const int width = data.size(3);
  const int pooled_height = out.size(2);
  const int pooled_width = out.size(3);
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  PSROIALIGNAVEPoolForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, stream >> >(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, sampling_ratio, bottom_rois, output_dim_, group_size_, top_data);
  PSROIALIGNAVEPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T* w1,
    T* w2,
    T* w3,
    T* w4,
    int* x_low,
    int* x_high,
    int* y_low,
    int* y_high,
    const int /*index*/ /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    *w1 = *w2 = *w3 = *w4 = 0.;
    *x_low = *x_high = *y_low = *y_high = -1;
    return;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  *y_low = static_cast<int>(y);
  *x_low = static_cast<int>(x);

  if (*y_low >= height - 1) {
    *y_high = *y_low = height - 1;
    y = (T)*y_low;
  } else {
    *y_high = *y_low + 1;
  }

  if (*x_low >= width - 1) {
    *x_high = *x_low = width - 1;
    x = (T)*x_low;
  } else {
    *x_high = *x_low + 1;
  }

  T ly = y - *y_low;
  T lx = x - *x_low;
  T hy = 1. - ly, hx = 1. - lx;

  *w1 = hy * hx, *w2 = hy * lx, *w3 = ly * hx, *w4 = ly * lx;

  return;
}

template <typename DType>
__global__ void PSROIALIGNAVEPoolBackwardAccKernel(
  const int count,
  const DType* top_diff,
  const int num_rois,
  const DType spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
  const int sampling_ratio,
  const int group_size,
  const int output_dim,
  DType* bottom_diff,
  const DType* bottom_rois) {
  CUDA_KERNEL_LOOP(index, count) {
    // The output is in order (n, ctop, ph, pw)
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int ctop = (index / pooled_width / pooled_height) % output_dim;
    int n = index / pooled_width / pooled_height / output_dim;

    // [start, end) interval for spatial sampling
    const DType* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    // DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
    // DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
    // DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale;
    // DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale;
    // Do not using rounding; this implementation detail is critical
    DType roi_start_w = offset_bottom_rois[1] * spatial_scale;
    DType roi_start_h = offset_bottom_rois[2] * spatial_scale;
    DType roi_end_w = offset_bottom_rois[3] * spatial_scale;
    DType roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force too small ROIs to be 1x1
    DType roi_width = max(roi_end_w - roi_start_w, (DType)1.);  // avoid 0
    DType roi_height = max(roi_end_h - roi_start_h, (DType)1.);

    // Compute w and h at bottom
    DType bin_size_h = static_cast<DType>(roi_height) / static_cast<DType>(pooled_height);
    DType bin_size_w = static_cast<DType>(roi_width) / static_cast<DType>(pooled_width);

    // int hstart = floor(static_cast<DType>(ph)* bin_size_h
    //   + roi_start_h);
    // int wstart = floor(static_cast<DType>(pw)* bin_size_w
    //   + roi_start_w);
    // int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h
    //   + roi_start_h);
    // int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w
    //   + roi_start_w);
    // // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart, 0), height);
    // hend = min(max(hend, 0), height);
    // wstart = min(max(wstart, 0), width);
    // wend = min(max(wend, 0), width);
    // bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;
    DType* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    // DType bin_area = (hend - hstart)*(wend - wstart);
    // DType diff_val = is_empty ? (DType)0. : top_diff[index] / bin_area;
    // for (int h = hstart; h < hend; ++h) {
    //   for (int w = wstart; w < wend; ++w) {
    //     int bottom_index = h*width + w;
    //     atomicAdd(offset_bottom_diff + bottom_index, diff_val);
    //   }
    // }
    // int top_offset = (n * channels + ctop) * pooled_height * pooled_width;
    // const DType* offset_top_diff = top_diff + top_offset;
    // const DType top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    const DType top_diff_this_bin = top_diff[index];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);
    // We do average (integral) pooling inside a bin
    const DType sample_count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const DType y = roi_start_h + ph * bin_size_h +
          static_cast<DType>(iy + .5f) * bin_size_h /
              static_cast<DType>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const DType x = roi_start_w + pw * bin_size_w +
            static_cast<DType>(ix + .5f) * bin_size_w /
                static_cast<DType>(roi_bin_grid_w);

        DType w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            &w1,
            &w2,
            &w3,
            &w4,
            &x_low,
            &x_high,
            &y_low,
            &y_high,
            index); //

        DType g1 = top_diff_this_bin * w1 / sample_count;
        DType g2 = top_diff_this_bin * w2 / sample_count;
        DType g3 = top_diff_this_bin * w3 / sample_count;
        DType g4 = top_diff_this_bin * w4 / sample_count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<DType>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<DType>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<DType>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<DType>(g4));
        }  // if
      }  // ix
    }  // iy          
  }
}


template<typename DType>
inline void PSROIALIGNAVEPoolBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
                            const Tensor<gpu, 4, DType> &out_grad,
                            const Tensor<gpu, 2, DType> &bbox,
                            const float spatial_scale,
                            const int sampling_ratio,
                            const int output_dim_,
                            const int group_size_) {
  // LOG(INFO) << "PSROIALIGNAVEPoolBackward";
  const DType *top_diff = out_grad.dptr_;
  const DType *bottom_rois = bbox.dptr_;
  DType *bottom_diff = in_grad.dptr_;
  const int count = out_grad.shape_.Size();
  const int num_rois = bbox.size(0);
  const int channels = in_grad.size(1);
  const int height = in_grad.size(2);
  const int width = in_grad.size(3);
  const int pooled_height = out_grad.size(2);
  const int pooled_width = out_grad.size(3);
  cudaStream_t stream = Stream<gpu>::GetStream(in_grad.stream_);
  PSROIALIGNAVEPoolBackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, stream >> >(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, sampling_ratio, group_size_, output_dim_, bottom_diff, bottom_rois);
  PSROIALIGNAVEPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace cuda

template<typename DType>
inline void PSROIALIGNAVEPoolForward(const Tensor<gpu, 4, DType> &out,
                           const Tensor<gpu, 4, DType> &data,
                           const Tensor<gpu, 2, DType> &bbox,
                           const float spatial_scale,
                           const int sampling_ratio,
                           const int output_dim_,
                           const int group_size_) {
  cuda::PSROIALIGNAVEPoolForward(out, data, bbox, spatial_scale, sampling_ratio,output_dim_, group_size_);
}

template<typename DType>
inline void PSROIALIGNAVEPoolBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
                            const Tensor<gpu, 4, DType> &out_grad,
                            const Tensor<gpu, 2, DType> &bbox,
                            const float spatial_scale,
                            const int sampling_ratio,
                            const int output_dim_,
                            const int group_size_) {
  cuda::PSROIALIGNAVEPoolBackwardAcc(in_grad, out_grad, bbox, spatial_scale, sampling_ratio, output_dim_, group_size_);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PSROIALIGNAVEPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PSROIALIGNAVEPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
