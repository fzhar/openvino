// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief VA API image classes
 * \file vaapi_images.h
 */

#pragma once

#include <algorithm>
#include <memory>

#include <va/va.h>
#include <va/va_drmcommon.h>

#include <opencv2/core.hpp>
#include <fourcc.h>

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
    VaApiImage(const std::shared_ptr<VaApiContext>& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface = VA_INVALID_ID);
    virtual ~VaApiImage();

    using Ptr = std::shared_ptr<VaApiImage>;

    void resizeTo(VaApiImage::Ptr dstImage, IMG_RESIZE_MODE resizeMode = RESIZE_FILL, bool hqResize=false);
    cv::Mat copyToMat(IMG_CONVERSION_TYPE convType = CONVERT_TO_BGR);

    uint32_t va_surface_id = VA_INVALID_ID;
    std::shared_ptr<VaApiContext> context = nullptr;

    FourCC format = FOURCC_NONE; // FourCC
    uint32_t width = 0 ;
    uint32_t height = 0;

  protected:
    VaApiImage(const VaApiImage& other) = delete;
    void destroyImage();

    VASurfaceID createVASurface();
};