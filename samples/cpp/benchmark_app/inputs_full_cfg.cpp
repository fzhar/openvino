// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inputs_full_cfg.hpp"
#include "samples\slog.hpp"

bool TestCfg::looks_like_image() const {
    if ((input_cfg.layout != "NCHW") && (input_cfg.layout != "NHWC") && (input_cfg.layout != "CHW") &&
        (input_cfg.layout != "HWC"))
        return false;
    // If data_shape is still empty, assume this is still an Image and tensor shape will be filled later
    return (data_shape.empty() || channels() == 3);
}
bool TestCfg::looks_like_image_info() const {
    if (input_cfg.layout != "NC")
        return false;
    return (channels() >= 2);
}
size_t TestCfg::width() const {
    return data_shape.at(ov::layout::width_idx(input_cfg.layout));
}
size_t TestCfg::height() const {
    return data_shape.at(ov::layout::height_idx(input_cfg.layout));
}
size_t TestCfg::channels() const {
    return data_shape.at(ov::layout::channels_idx(input_cfg.layout));
}
size_t TestCfg::batch() const {
    return data_shape.at(ov::layout::batch_idx(input_cfg.layout));
}
size_t TestCfg::depth() const {
    return data_shape.at(ov::layout::depth_idx(input_cfg.layout));
}

bool TestCfg::has_batch() const {
    return ov::layout::has_batch(input_cfg.layout));
}

std::vector<size_t> InputsFullCfg::get_batch_sizes() {
    std::vector<size_t> batches(get_tests_count());
    for (auto& input_cfg : *this) {
        for (int test_num = 0; test_num < get_tests_count(); test_num++) {
            if (input_cfg.second.tests[test_num].has_batch()) {
                if (batches[test_num] == 0)
                    batches[test_num] = input_cfg.second.tests[test_num].batch();
                else if (batches[test_num] != input_cfg.second.tests[test_num].batch())
                    throw std::logic_error("Can't deterimine batch size: batch is "
                                           "different for different inputs!");
            }
        }
        for (int test_num = 0; test_num < get_tests_count(); test_num++) {
            if (batches[test_num] == 0) {
                slog::warn << "No batch dimension wa/s found at any input for test " + std::to_string(test_num) +
                                  ", asssuming batch to be 1. Beware: this might affect "
                                  "FPS calculation."
                           << slog::endl;
                batches[test_num] = 1;
            }
        }
    }
    return batches;
}
