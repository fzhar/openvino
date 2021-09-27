// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief gStreamer-based video decoder class
 * \file gst_vaapi_decoder.h
 */

#pragma once

#include <string>

#include <gst/gst.h>
#include <glib-object.h>
#include <gst/app/gstappsink.h>
#include <gst/allocators/allocators.h>
#include <gst/gststructure.h>
#include <gst/gstquery.h>
#include <gst/video/video.h>
#include "vaapi_context.h"

#include "vaapi_image.h"

class GstVaApiDecoder
{
public:
    GstVaApiDecoder(int outWidth, int outHeight);
    ~GstVaApiDecoder();

public:
    void open(const std::string& filename, bool sync = false);
    void play();
    bool read(std::shared_ptr<VaApiImage>& image);
    void close();
    double getFPS(){ return fps;}

private:
    std::shared_ptr<VaApiImage>  CreateImage(GstSample* sampleRead, GstMapFlags map_flags);
    std::unique_ptr<VaApiImage> bufferToImage(GstBuffer *buffer);

    VaApiContext::Ptr vaContext;
    std::string filename_;

    GstElement* pipeline_;
    GstElement* file_source_;
    GstElement* demux_;
    GstElement* parser_;
    GstElement* dec_;
    GstElement* capsfilter_;
    GstElement* postproc_;
    GstElement* capsfilter2_;
    
    GstElement* queue_;
    GstElement* app_sink_;

    GstVideoInfo* video_info_;
    double fps;

    size_t outWidth;
    size_t outHeight;
};
