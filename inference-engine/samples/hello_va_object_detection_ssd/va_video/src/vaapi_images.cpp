// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vaapi_images.h"
#include <iostream>
#include <unistd.h>
#include <opencv2/imgproc.hpp>
#include <vaapi_context.h>

VASurfaceID VaApiImage::createVASurface() {
    return VaApiContext::createSurface(context->display(),width,height,format);
}

VaApiImage::VaApiImage(const VaApiContext::Ptr& context, uint32_t width, uint32_t height, FourCC format, uint32_t va_surface) {
    this->width = width;
    this->height = height;
    this->format = format;
    this->context = context;
    this->va_surface_id = va_surface == VA_INVALID_ID ? createVASurface() : va_surface;
}


VaApiImage::~VaApiImage() {
    if (va_surface_id != VA_INVALID_ID) {
        try {
            VA_CALL(vaDestroySurfaces(context->display(), (uint32_t *)&va_surface_id, 1));
        } catch (const std::exception &e) {
            std::string error_message = std::string("VA surface destroying failed with exception: ") + e.what();
            std::cout << error_message.c_str() << std::endl;
        }
    }
}

cv::Mat VaApiImage::copyToMat(IMG_CONVERSION_TYPE convType) {

    VAImage mappedImage;
    void *pData = nullptr;
    cv::Mat outMat;
    cv::Mat mappedMat;

    //--- Mapping image
    VA_CALL(vaDeriveImage(context->display(), va_surface_id, &mappedImage))
    VA_CALL(vaMapBuffer(context->display(), mappedImage.buf, &pData))
    //--- Copying data to Mat. Only NV12/I420 formats are supported now
    switch(format) {
        case FOURCC_NV12:
        case FOURCC_I420:
        {
            // NV12 image might be aligned by height to 16 or 32, so during conversion it should use actual width first
            // (to avoid gaps between Y and UV), and then it should be clipped to actual height
            int alignedRows = mappedImage.offsets[1] / mappedImage.pitches[0];
            mappedMat = cv::Mat(alignedRows*3/2,mappedImage.width,CV_8UC1,pData,mappedImage.pitches[0]);
            break;
        }
        default:
            throw std::invalid_argument("VAApiImage Map: non-supported FOURCC encountered");
    }

    //--- Converting image
    switch(convType) {
        case CONVERT_TO_RGB:
            cv::cvtColor(mappedMat,outMat,format == FOURCC_NV12 ? cv::COLOR_YUV2RGB_NV12 : cv::COLOR_YUV2RGB);
            break;
        case CONVERT_TO_BGR:
            cv::cvtColor(mappedMat,outMat,format == FOURCC_NV12 ? cv::COLOR_YUV2BGR_NV12 : cv::COLOR_YUV2BGR);
            break;
        default:
            mappedMat.copyTo(outMat);
            break;
    }
    try {
        VA_CALL(vaUnmapBuffer(context->display(), mappedImage.buf))
        VA_CALL(vaDestroyImage(context->display(), mappedImage.image_id));
    } catch (const std::exception &e) {
        std::string error_message =
            std::string("VA buffer unmapping (destroying) failed with exception: ") + e.what();
        std::cout << error_message.c_str() << std::endl;
    }

    return outMat(cv::Rect(0,0,mappedImage.width,mappedImage.height));
}

void VaApiImage::resizeTo(VaApiImage::Ptr dstImage, IMG_RESIZE_MODE resizeMode, bool hqResize) {
    if(context->display() != dstImage->context->display() || context->contextId() != dstImage->context->contextId())
    {
        throw std::invalid_argument("resizeTo: (context, display) of the source and destination images should be the same");
    }

    VAProcPipelineParameterBuffer pipelineParam = VAProcPipelineParameterBuffer();
    pipelineParam.surface = va_surface_id;
    VARectangle surface_region = {.x = 0,
                                  .y = 0,
                                  .width = (uint16_t)this->width,
                                  .height = (uint16_t)this->height};
    if (surface_region.width > 0 && surface_region.height > 0)
        pipelineParam.surface_region = &surface_region;

    pipelineParam.filter_flags = hqResize ? VA_FILTER_SCALING_HQ : VA_FILTER_SCALING_DEFAULT;

    VABufferID pipelineParamBufId = VA_INVALID_ID;
    VA_CALL(vaCreateBuffer(context->display(), context->contextId(), VAProcPipelineParameterBufferType,
                           sizeof(pipelineParam), 1, &pipelineParam, &pipelineParamBufId));

    VA_CALL(vaBeginPicture(context->display(), context->contextId(), dstImage->va_surface_id))

    VA_CALL(vaRenderPicture(context->display(), context->contextId(), &pipelineParamBufId, 1))

    VA_CALL(vaEndPicture(context->display(), context->contextId()))

    VA_CALL(vaDestroyBuffer(context->display(), pipelineParamBufId))
}
