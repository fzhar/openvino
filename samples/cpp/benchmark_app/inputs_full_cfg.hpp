// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <map>
#include <openvino/openvino.hpp>

struct InputCfg;

struct TestCfg {
    enum InputDataType {
        RND,
        IMAGE,
        IMAGE_INFO,
        BINARY
    };
    TestCfg(const InputCfg& input_cfg, const ov::Shape& data_shape = ov::Shape(), InputDataType data_type = RND)
        : data_shape(data_shape),
        data_type(data_type),
        p_input_cfg(&input_cfg) {};

    std::vector<std::string> filenames;
    ov::Shape data_shape;
    InputDataType data_type;

    size_t width() const;
    size_t height() const;
    size_t channels() const;
    size_t batch() const;
    size_t depth() const;

    bool has_batch() const;
    const InputCfg& input_cfg() const {
        return *p_input_cfg;
    }

private:
    const InputCfg* p_input_cfg;
};

struct InputCfg {
    ov::element::Type type;
    ov::PartialShape partial_shape;
    ov::Layout layout;
    std::vector<float> scale;
    std::vector<float> mean;

    std::vector<TestCfg> tests;

    bool has_batch() const;
    bool looks_like_image() const;
    bool looks_like_image_info() const;
};

class InputsFullCfg : public std::map<std::string, InputCfg> {
public:
    size_t get_tests_count() {
        return size() > 0 ? this->at(0).tests.size() : 0;
    }
    std::vector<size_t> get_batch_sizes();
};