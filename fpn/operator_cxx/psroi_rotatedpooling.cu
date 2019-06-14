/*!
 * Copyright (c) 2017 by Contributors
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file psroi_pooling.cu
 * \brief psroi pooling operator
 * \author Yi Li, Tairui Chen, Guodong Zhang, Haozhi Qi, Jifeng Dai
 * modified by Jian Ding
*/
#include "./psroi_rotatedpooling-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

#define PSROIROTATEDPOOLING_CUDA_CHECK(condition) \
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

template <typename DType>
__global__ void PSROIROTATEDPoolForwardKernel(
  const int count,
  const DType* bottom_data,
  const DType spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
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
    const DType* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];
    DType roi_xc = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
    DType roi_yc = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
    DType roi_w = static_cast<DType>(round(offset_bottom_rois[3])) * spatial_scale;
    DType roi_h = static_cast<DType>(round(offset_bottom_rois[4])) * spatial_scale;
    DType Theta = static_cast<DType>(offset_bottom_rois[5]);

    DType cosTheta = cos(Theta);
    DType sinTheta = sin(Theta);

    // Force too small ROIs to be 1x1
    DType roi_width = max(roi_w, 1.);  // avoid 0
    DType roi_height = max(roi_h, 1.);

    // Compute w and h at bottom
    DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
    DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

    int hstart = floor(static_cast<DType>(ph) * bin_size_h);
    int wstart = floor(static_cast<DType>(pw)* bin_size_w);
    int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h);
    int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w);
    // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart, 0), roi_h);
    // hend = min(max(hend, 0), roi_h);
    // wstart = min(max(wstart, 0), roi_w);
    // wend = min(max(wend, 0), roi_w);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;

    const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    DType out_sum = 0;
    float half_w = (float)(roi_w)/2.0;
    float half_h = (float)(roi_h)/2.0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        float xx = cosTheta*((float)(w) - half_w) - sinTheta*((float)(h) - half_h) + roi_xc;
        float yy = sinTheta*((float)(w) - half_w) + cosTheta*((float)(h) - half_h) + roi_yc; 
        int xint = (int)(round(xx));
        int yint = (int)(round(yy));
        
        if (xint >= width || xint < 0 || yint>=height || yint < 0)
          continue;

        int bottom_index = yint*width + xint;
        out_sum += offset_bottom_data[bottom_index];
      }
    }

    DType bin_area = (hend - hstart)*(wend - wstart);
    top_data[index] = is_empty? (DType)0. : out_sum/bin_area;
  }
}

template<typename DType>
inline void PSROIROTATEDPoolForward(const Tensor<gpu, 4, DType> &out,
                           const Tensor<gpu, 4, DType> &data,
                           const Tensor<gpu, 2, DType> &bbox,
                           const float spatial_scale,
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
  PSROIROTATEDPoolForwardKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, stream >> >(
      count, bottom_data, spatial_scale, channels, height, width,
      pooled_height, pooled_width, bottom_rois, output_dim_, group_size_, top_data);
  PSROIROTATEDPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}


template <typename DType>
__global__ void PSROIROTATEDPoolBackwardAccKernel(
  const int count,
  const DType* top_diff,
  const int num_rois,
  const DType spatial_scale,
  const int channels,
  const int height, const int width,
  const int pooled_height, const int pooled_width,
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
    const DType* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];
    DType roi_xc = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
    DType roi_yc = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
    DType roi_w = static_cast<DType>(round(offset_bottom_rois[3])) * spatial_scale;
    DType roi_h = static_cast<DType>(round(offset_bottom_rois[4])) * spatial_scale;
    DType Theta = static_cast<DType>(offset_bottom_rois[5]);

    DType cosTheta = cos(Theta);
    DType sinTheta = sin(Theta);

    // Force too small ROIs to be 1x1
    DType roi_width = max(roi_w, 1.0);  // avoid 0
    DType roi_height = max(roi_h, 1.0);

    // Compute w and h at bottom
    DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
    DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

    int hstart = floor(static_cast<DType>(ph)* bin_size_h);
    int wstart = floor(static_cast<DType>(pw)* bin_size_w);
    int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h);
    int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w);
    // Add roi offsets and clip to input boundaries
    // hstart = min(max(hstart, 0), roi_h);
    // hend = min(max(hend, 0), roi_h);
    // wstart = min(max(wstart, 0), roi_w);
    // wend = min(max(wend, 0), roi_w);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Compute c at bottom
    int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);
    int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);
    int c = (ctop*group_size + gh)*group_size + gw;
    DType* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
    DType bin_area = (hend - hstart)*(wend - wstart);
    DType diff_val = is_empty ? (DType)0. : top_diff[index] / bin_area;

    float half_w = (float)(roi_w)/2.0;
    float half_h = (float)(roi_h)/2.0;

    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        float xx = cosTheta*((float)(w)-half_w) - sinTheta*((float)(h)-half_h) + roi_xc;
        float yy = sinTheta*((float)(w)-half_w) + cosTheta*((float)(h)-half_h) + roi_yc;
        int xint = (int)(round(xx));
        int yint = (int)(round(yy));

        if (xint>=width || xint<0 || yint>=height || yint<0)
          continue;

        int bottom_index = yint*width + xint;
        atomicAdd(offset_bottom_diff + bottom_index, diff_val);
      }
    }
  }
}


template<typename DType>
inline void PSROIROTATEDPoolBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
                            const Tensor<gpu, 4, DType> &out_grad,
                            const Tensor<gpu, 2, DType> &bbox,
                            const float spatial_scale,
                            const int output_dim_,
                            const int group_size_) {
  // LOG(INFO) << "PSROIROTATEDPoolBackward";
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
  PSROIROTATEDPoolBackwardAccKernel<DType> << <mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, stream >> >(
      count, top_diff, num_rois, spatial_scale, channels, height, width,
      pooled_height, pooled_width, group_size_, output_dim_, bottom_diff, bottom_rois);
  PSROIROTATEDPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace cuda

template<typename DType>
inline void PSROIROTATEDPoolForward(const Tensor<gpu, 4, DType> &out,
                           const Tensor<gpu, 4, DType> &data,
                           const Tensor<gpu, 2, DType> &bbox,
                           const float spatial_scale,
                           const int output_dim_,
                           const int group_size_) {
  cuda::PSROIROTATEDPoolForward(out, data, bbox, spatial_scale, output_dim_, group_size_);
}

template<typename DType>
inline void PSROIROTATEDPoolBackwardAcc(const Tensor<gpu, 4, DType> &in_grad,
                            const Tensor<gpu, 4, DType> &out_grad,
                            const Tensor<gpu, 2, DType> &bbox,
                            const float spatial_scale,
                            const int output_dim_,
                            const int group_size_) {
  cuda::PSROIROTATEDPoolBackwardAcc(in_grad, out_grad, bbox, spatial_scale, output_dim_, group_size_);
}

}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PSROIROTATEDPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PSROIROTATEDPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
