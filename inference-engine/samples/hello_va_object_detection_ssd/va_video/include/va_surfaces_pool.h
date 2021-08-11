// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief VA surfaces pool class
 * \file va_surfaces_pool.h
 */

#pragma once

#include <unordered_map>
#include <future>
#include "fourcc.h"

class VaApiContext;
class VaApiImage;

class VaSurfacesPool {
  public:
    VASurfaceID acquire(uint16_t width, uint16_t height, FourCC fourcc);
    void release(const VaApiImage& img);
    void waitForCompletion();
    VaSurfacesPool() : display(nullptr) {}
    VaSurfacesPool(VADisplay display) : display(display) {}
    ~VaSurfacesPool();
  private:
    using Element = std::pair<VASurfaceID, bool>; // second is true if image is in use
    std::unordered_multimap<uint64_t, Element> images;
    std::condition_variable _free_image_condition_variable;
    std::mutex mtx;

    uint64_t calcKey(uint16_t width, uint16_t height, FourCC fourcc) {
        return static_cast<uint64_t>(fourcc) |
            ((static_cast<uint64_t>(width) & 0xFFFF)<<32) | ((static_cast<uint64_t>(height) & 0xFFFF)<<48);
    }

    VADisplay display;
};
