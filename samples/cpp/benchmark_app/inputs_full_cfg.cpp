// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inputs_full_cfg.hpp"
#include "samples\slog.hpp"

bool InputCfg::looks_like_image() const {
    if ((layout != "NCHW") && (layout != "NHWC") && (layout != "CHW") &&
        (layout != "HWC"))
        return false;

    size_t index = -1;
    try {
        index = layout.get_index_by_name("N");
    } catch (...) {
    }

    // If shape is does not contain static channles value, assume this is still an Image
    return index >= 0 && (partial_shape[index].is_dynamic() || partial_shape[index].get_min_length() == 3);
}

bool InputCfg::looks_like_image_info() const {
    if (layout != "NC")
        return false;
    size_t index = -1;
    try {
        index = layout.get_index_by_name("N");
    } catch (...) {
    }
    return index >= 0 && (partial_shape[index].is_dynamic() || partial_shape[index].get_min_length() >= 2);
}
size_t TestCfg::width() const {
    return data_shape.at(ov::layout::width_idx(input_cfg().layout));
}
size_t TestCfg::height() const {
    return data_shape.at(ov::layout::height_idx(input_cfg().layout));
}
size_t TestCfg::channels() const {
    return data_shape.at(ov::layout::channels_idx(input_cfg().layout));
}
size_t TestCfg::batch() const {
    return data_shape.at(ov::layout::batch_idx(input_cfg().layout));
}
size_t TestCfg::depth() const {
    return data_shape.at(ov::layout::depth_idx(input_cfg().layout));
}

bool TestCfg::has_batch() const {
    return input_cfg().has_batch();
}

bool InputCfg::has_batch() const {
    return ov::layout::has_batch(layout);
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
                slog::warn << "No batch dimension was found at any input for test " + std::to_string(test_num) +
                                  ", asssuming batch to be 1. Beware: this might affect "
                                  "FPS calculation."
                           << slog::endl;
                batches[test_num] = 1;
            }
        }
    }
    return batches;
}
