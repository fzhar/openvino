// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief VA API conext class (VA display, contexts, config and surfaces pool in one place)
 * \file vaapi_context.h
 */

#pragma once

#include "vaapi_utils.h"

#include <functional>
#include <stdexcept>

#include <va/va.h>
#include <gpu/gpu_context_api_va.hpp>

#include <cstdint>
#include <type_traits>
#include <memory>
#include <fourcc.h>

class VaApiContext
{
  private:
    VADisplay vaDisplay = nullptr;
    VAConfigID vaConfig = VA_INVALID_ID;
    VAContextID vaContextId = VA_INVALID_ID;
    int driFileDescriptor = 0;
    bool isOwningVaDisplay = false;
    InferenceEngine::gpu::VAContext::Ptr gpuSharedContext = nullptr;

  public:
    using Ptr=std::shared_ptr<VaApiContext>;
    VaApiContext(VADisplay display = nullptr);
    VaApiContext(VADisplay display, InferenceEngine::Core& coreForSharedContext);
    VaApiContext(InferenceEngine::Core& coreForSharedContext);

    ~VaApiContext();

    void createSharedContext(InferenceEngine::Core& core);

    VAContextID contextId() {
      return vaContextId;
    }

    VADisplay display() {
      return vaDisplay;
    }

    InferenceEngine::gpu::VAContext::Ptr sharedContext() {
      return gpuSharedContext;
    }

    static VASurfaceID createSurface(VADisplay display, uint16_t width, uint16_t height, FourCC format);

  private:
    void create(VADisplay display);
    void close();

    static void messageErrorCallback(void *, const char *message);
    static void messageInfoCallback(void *, const char *message);
};
