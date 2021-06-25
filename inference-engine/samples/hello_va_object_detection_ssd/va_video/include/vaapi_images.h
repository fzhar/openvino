/*
// Copyright (C) 2018-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include <algorithm>
#include <memory>

#include <va/va.h>
#include <va/va_drmcommon.h>

#include <opencv2/core.hpp>
#include "vaapi_image_pool.h"

enum IMG_RESIZE_MODE {
    RESIZE_FILL,
    RESIZE_KEEP_ASPECT,
    RESIZE_KEEP_ASPECT_LETTERBOX
};

enum IMG_CONVERSION_TYPE {
    CONVERT_NONE,
    CONVERT_TO_RGB,
    CONVERT_TO_BGR,
};

class VaApiContext;

class VaApiImage{
  public:
    VaApiImage() {};
    VaApiImage(const std::shared_ptr<VaApiContext>& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface = VA_INVALID_ID, bool autoDestroySurface=true);
    virtual ~VaApiImage() { if(autoDestroySurface) destroyImage(); }

    using Ptr = std::shared_ptr<VaApiImage>;

    VaApiImage::Ptr cloneToAnotherContext(const std::shared_ptr<VaApiContext>& newContext);
    void resizeTo(VaApiImage::Ptr dstImage, IMG_RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize=false);
    VaApiImage::Ptr resizeUsingPooledSurface(uint16_t width, uint16_t height, IMG_RESIZE_MODE resizeMode, bool hqResize);

    cv::Mat copyToMat(IMG_CONVERSION_TYPE convType = CONVERT_TO_BGR);

    uint32_t va_surface_id = VA_INVALID_ID;
    std::shared_ptr<VaApiContext> context = nullptr;

    FourCC format = FOURCC_NONE; // FourCC
    uint32_t width = 0 ;
    uint32_t height = 0;

  protected:
    bool autoDestroySurface;

    VaApiImage(const VaApiImage& other) = delete;
    void destroyImage();

    VASurfaceID createVASurface();
};

class VaPooledImage : public VaApiImage{
  public:
    using Ptr = std::shared_ptr<VaPooledImage>;
    VaPooledImage(const std::shared_ptr<VaApiContext>& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface) :
      VaApiImage(context, width, height, format, va_surface, false) {}
    ~VaPooledImage() override;
};
