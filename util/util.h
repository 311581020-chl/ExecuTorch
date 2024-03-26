/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/platform/log.h>
#ifdef USE_ATEN_LIB
#include <ATen/ATen.h> // @manual=//caffe2/aten:ATen-core
#endif

struct __attribute__((packed)) Image {
    uint8_t label;
    // std::vector<uint8_t> pixels;
    uint8_t pixels[3072];
};

namespace torch {
namespace executor {
namespace util {

using namespace exec_aten;

// This macro defines all the scalar types that we currently support to fill a
// tensor. FillOnes() is a quick and dirty util that allows us to quickly
// initialize tensors and run a model.
#define EX_SCALAR_TYPES_SUPPORTED_BY_FILL(_fill_case) \
  _fill_case(uint8_t, Byte) /* 0 */                   \
      _fill_case(int8_t, Char) /* 1 */                \
      _fill_case(int16_t, Short) /* 2 */              \
      _fill_case(int, Int) /* 3 */                    \
      _fill_case(int64_t, Long) /* 4 */               \
      _fill_case(float, Float) /* 6 */                \
      _fill_case(double, Double) /* 7 */              \
      _fill_case(bool, Bool) /* 11 */

#define FILL_CASE(T, n)                                \
  case (ScalarType::n):                                \
    std::fill(                                         \
        tensor.mutable_data_ptr<T>(),                  \
        tensor.mutable_data_ptr<T>() + tensor.numel(), \
        1);                                            \
    break;

#ifndef USE_ATEN_LIB
inline void FillOnes(Tensor tensor) {
  switch (tensor.scalar_type()) {
    EX_SCALAR_TYPES_SUPPORTED_BY_FILL(FILL_CASE)
    default:
      ET_CHECK_MSG(false, "Scalar type is not supported by fill.");
  }
}
#endif

/*  TODO: 
 *    Resize the image from 3*32*32 to 3*224*224 by bilinear interpolation
 *  
 *  Params:
 *    inputPixels: an array with size 3*32*32 whose value are between 0 and 255, representing original image
 *    outputPixels: an array with size 3*224*224, representing resized image
 * 
 *  Note: 
 *    1. you can use OpenMP to speed up if you want
 *    2. you can refer to https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/interpolation/bilinear-filtering.html for more information
 */ 
float bilinear(
   const float tx, 
   const float ty, 
   float c00, 
   float c10,
   float c01,
   float c11)
{
#if 1
    float  a = c00 * (1 - tx) + c10 * tx;
    float  b = c01 * (1 - tx) + c11 * tx;
    return a * (1 - ty) + b * ty;
#else
    return (1 - tx) * (1 - ty) * c00 + 
        tx * (1 - ty) * c10 +
        (1 - tx) * ty * c01 +
        tx * ty * c11;
#endif
}

inline void ResizeImage(uint8_t* inputPixels, float* outputPixels) {
    const int inputWidth = 32;
    const int inputHeight = 32;
    const int outputWidth = 224;
    const int outputHeight = 224;

    const int channels = 3;

    /* Your CODE starts here */

    for (size_t c = 0; c < channels; c++)
    {
      for (size_t j = 0; j < outputHeight; j++)
      {
        for (size_t i = 0; i < outputWidth; i++)
        {
          float gx = i / float(outputWidth) * inputWidth; // be careful to interpolate boundaries
          float gy = j / float(outputHeight) * inputHeight; // be careful to interpolate boundaries
          int gxi = int(gx);
          int gyi = int(gy);
          float c00 = inputPixels[c * inputHeight * inputWidth + gyi * inputWidth + gxi];
          float c10 = inputPixels[c * inputHeight * inputWidth + gyi * inputWidth + (gxi + 1)];
          float c01 = inputPixels[c * inputHeight * inputWidth + (gyi + 1) * inputWidth + gxi];
          float c11 = inputPixels[c * inputHeight * inputWidth + (gyi + 1) * inputWidth + (gxi + 1)];
          *(outputPixels++) = bilinear(gx - gxi, gy - gyi, c00, c10, c01, c11);
        }
        
      }
      
    }

}

/*  TODO: 
 *    Divide the value of image by 255, shrink the value of image to the range between 0 and 1
 *  
 *  Params:
 *    imgPixels: an array with size 3*224*224, representing an image
 *    num_pix: number of pixels
 * 
 *  Note: you can use OpenMP to speed up if you want
 */
inline void ToTensor(float* imgPixels, int num_pix){

    /* Your CODE starts here */
    for (size_t i = 0; i < num_pix; i++)
    {
      // std::cout << imgPixels[i] << std::endl;
      imgPixels[i] = imgPixels[i] / 255;
      // std::cout << imgPixels[i] << std::endl;
    }
    
}

/*  TODO: 
 *    Normalize the image with given mean and std
 *  
 *  Params:
 *    imgTensor: an array with size 3*224*224, representing the tensor of an image 
 *    n_pix: number of pixels of the image
 *    mean: array of value indicating the mean of each channel
 *    std: array of value indicating the std of each channel
 *    n_ch: number of channel
 * 
 *  Note: you can use OpenMP to speed up if you want
 */
inline void Normalize(float* imgTensor, int n_pix, float* mean, float* std, int n_ch){

    int pix_per_ch = n_pix / n_ch;

    /* Your CODE starts here */

    for (size_t c = 0; c < n_ch; c++)
    {
      for (size_t i = 0; i < pix_per_ch; i++)
      {
        imgTensor[c * pix_per_ch + i] = (imgTensor[c * pix_per_ch + i] - mean[c]) / std[c];
      }
      
    }
    

}

/*  TODO: 
 *    Do Data pre-processing on given image and load it to tensor object
 *    Data pre-processing includes Resize, ToTensor, and Normalize
 *  
 *  Params:
 *    tensor: Tensor object, refer to executorch/runtime/core/portable_type/tensor.h  
 *    image: a struct with label and pixel value included
 *    mean: array of value indicating the mean of each channel
 *    std: array of value indicating the std of each channel
 */
inline void LoadImage(Tensor tensor, Image& image, float* mean, float* std) {
    float* img_ptr = tensor.mutable_data_ptr<float>();
    int n_pix = tensor.numel();

    /* Your CODE starts here */

    ResizeImage(image.pixels, img_ptr);
    // int n_pc = tensor.size(1) * tensor.size(2);
    ToTensor(img_ptr, n_pix);
    Normalize(img_ptr, n_pix, mean, std, tensor.size(0));

}

// Read image from test_batch.bin, you can write on your own
inline Image ReadImage(int fd) {
    Image img;
    
    ssize_t nbytes = read(fd, &img, sizeof(Image));
    if (nbytes < 0)
      ET_LOG(Info, "Error reading image.");
    
    return img;
}

/**
 * Allocates input tensors for the provided Method, filling them with ones.
 *
 * @param[in] method The Method that owns the inputs to prepare.
 * @returns An array of pointers that must be passed to `FreeInputs()` after
 *     the Method is no longer needed.
 */
inline exec_aten::ArrayRef<void*> PrepareInputTensors(Method& method, Image& img) {
  auto method_meta = method.method_meta();
  size_t input_size = method.inputs_size();
  size_t num_allocated = 0;
  void** inputs = (void**)malloc(input_size * sizeof(void*));

  for (size_t i = 0; i < input_size; i++) {
    if (*method_meta.input_tag(i) != Tag::Tensor) {
      ET_LOG(Info, "input %zu is not a tensor, skipping", i);
      continue;
    }

    // Tensor Input. Grab meta data and allocate buffer
    auto tensor_meta = method_meta.input_tensor_meta(i);
    inputs[num_allocated++] = malloc(tensor_meta->nbytes());

#ifdef USE_ATEN_LIB
    std::vector<int64_t> at_tensor_sizes;
    for (auto s : tensor_meta->sizes()) {
      at_tensor_sizes.push_back(s);
    }
    at::Tensor t = at::from_blob(
        inputs[num_allocated - 1],
        at_tensor_sizes,
        at::TensorOptions(tensor_meta->scalar_type()));
    t.fill_(1.0f);

#else // Portable Tensor
    // The only memory that needs to persist after set_input is called is the
    // data ptr of the input tensor, and that is only if the Method did not
    // memory plan buffer space for the inputs and instead is expecting the user
    // to provide them. Meta data like sizes and dim order are used to ensure
    // the input aligns with the values expected by the plan, but references to
    // them are not held onto.

    TensorImpl::SizesType* sizes = static_cast<TensorImpl::SizesType*>(
        malloc(sizeof(TensorImpl::SizesType) * tensor_meta->sizes().size()));
    TensorImpl::DimOrderType* dim_order =
        static_cast<TensorImpl::DimOrderType*>(malloc(
            sizeof(TensorImpl::DimOrderType) *
            tensor_meta->dim_order().size()));

    for (size_t size_idx = 0; size_idx < tensor_meta->sizes().size();
         size_idx++) {
      sizes[size_idx] = tensor_meta->sizes()[size_idx];
    }
    for (size_t dim_idx = 0; dim_idx < tensor_meta->dim_order().size();
         dim_idx++) {
      dim_order[dim_idx] = tensor_meta->dim_order()[dim_idx];
    }

    TensorImpl impl = TensorImpl(
        tensor_meta->scalar_type(),
        tensor_meta->sizes().size(),
        sizes,
        inputs[num_allocated - 1],
        dim_order);
    Tensor t(&impl);

    // Provide mean and std for data pre-processing
    float mean[3] = {0.485, 0.456, 0.406};
    float std[3] = {0.229, 0.224, 0.225};
    LoadImage(t, img, mean, std);

#endif
    auto error = method.set_input(t, i);
    ET_CHECK_MSG(
        error == Error::Ok,
        "Error: 0x%" PRIx32 " setting input %zu.",
        error,
        i);
#ifndef USE_ATEN_LIB // Portable Tensor
    free(sizes);
    free(dim_order);
#endif
  }

  return {inputs, num_allocated};
  
}

/**
 * Frees memory that was allocated by `PrepareInputTensors()`.
 */
inline void FreeInputs(exec_aten::ArrayRef<void*> inputs) {
  for (size_t i = 0; i < inputs.size(); i++) {
    free(inputs[i]);
  }
  free((void*)inputs.data());
}

#undef FILL_VALUE
#undef EX_SCALAR_TYPES_SUPPORTED_BY_FILL

} // namespace util
} // namespace executor
} // namespace torch
