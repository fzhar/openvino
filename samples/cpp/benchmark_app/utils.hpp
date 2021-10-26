// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ngraph/partial_shape.hpp"

namespace benchmark_app {
struct InputInfo {
    InferenceEngine::Precision precision;
    ngraph::PartialShape partialShape;
    InferenceEngine::SizeVector tensorShape;
    std::string layout;
    InferenceEngine::Layout _layout;
    std::vector<float> scale;
    std::vector<float> mean;
    bool isImage() const;
    bool isImageInfo() const;
    size_t getDimentionByLayout(char character) const;
    size_t width() const;
    size_t height() const;
    size_t channels() const;
    size_t batch() const;
    size_t depth() const;
};
using InputsInfo = std::map<std::string, InputInfo>;
using PartialShapes = std::map<std::string, ngraph::PartialShape>;
}  // namespace benchmark_app

std::vector<std::string> parseDevices(const std::string& device_string);
uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device);
std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
                                                               const std::string& values_string);
std::string getShapeString(const InferenceEngine::SizeVector& shape);
std::string getShapesString(const benchmark_app::PartialShapes& shapes);
std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes);
size_t getBatchSize(const benchmark_app::InputsInfo& inputs_info);
std::vector<std::string> split(const std::string& s, char delim);
std::map<std::string, std::vector<float>> parseScaleOrMean(const std::string& scale_mean,
                                                           const benchmark_app::InputsInfo& inputs_info);
std::vector<ngraph::Dimension> parsePartialShape(const std::string& partial_shape);
InferenceEngine::SizeVector parseTensorShape(const std::string& tensor_shape);

template <typename T>
std::map<std::string, std::vector<std::string>> parseInputParameters(const std::string parameter_string,
                                                                     const std::map<std::string, T>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::vector<std::string>> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    auto input_name = search_string.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        if (start_pos)
            input_name = search_string.substr(0, start_pos - 1);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[input_name].push_back(input_value);
        } else {
            for (auto& item : input_info) {
                return_value[item.first].push_back(input_value);
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',' && search_string.front() != '[')
            break;
        if (search_string.front() == ',')
            search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}

template <typename T>
std::vector<benchmark_app::InputsInfo> getInputsInfo(const std::string& shape_string,
                                                     const std::string& layout_string,
                                                     const size_t batch_size,
                                                     const std::string& tensors_shape_string,
                                                     const std::string& scale_string,
                                                     const std::string& mean_string,
                                                     const std::map<std::string, T>& input_info,
                                                     bool& reshape_required) {
    std::map<std::string, std::vector<std::string>> shape_map = parseInputParameters(shape_string, input_info);
    std::map<std::string, std::vector<std::string>> tensors_shape_map =
        parseInputParameters(tensors_shape_string, input_info);
    std::map<std::string, std::vector<std::string>> layout_map = parseInputParameters(layout_string, input_info);

    size_t min_size = 1, max_size = 1;
    if (!tensors_shape_map.empty()) {
        min_size = std::min_element(tensors_shape_map.begin(),
                                    tensors_shape_map.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size() && a.second.size() != 1;
                                    })
                       ->second.size();

        max_size = std::max_element(tensors_shape_map.begin(),
                                    tensors_shape_map.end(),
                                    [](std::pair<std::string, std::vector<std::string>> a,
                                       std::pair<std::string, std::vector<std::string>> b) {
                                        return a.second.size() < b.second.size();
                                    })
                       ->second.size();
        if (min_size != max_size) {
            throw std::logic_error("Number of shapes for all inputs must be the same (except inputs with 1 shape).");
        }
    }

    reshape_required = false;

    std::vector<benchmark_app::InputsInfo> info_maps;

    for (size_t i = 0; i < min_size; ++i) {
        benchmark_app::InputsInfo info_map;
        for (auto& item : input_info) {
            benchmark_app::InputInfo info;
            auto name = item.first;
            auto descriptor = item.second->getTensorDesc();
            // Precision
            info.precision = descriptor.getPrecision();
            // Partial Shape
            if (shape_map.count(name)) {
                std::vector<ngraph::Dimension> parsed_shape;
                if (shape_map.at(name).size() > 1) {
                    throw std::logic_error(
                        "shape command line parameter doesn't support multiple shapes for one input.");
                }
                info.partialShape = parsePartialShape(shape_map.at(name)[0]);
                reshape_required = true;
            } else {
                info.partialShape = item.second->getPartialShape();
            }

            if (info.partialShape.is_dynamic() && info.isImage()) {
                throw std::logic_error(
                    "benchmark_app supports only binary and random data as input for dynamic models at this moment.");
            }

            // Tensor Shape
            if (info.partialShape.is_dynamic() && tensors_shape_map.count(name)) {
                info.tensorShape = parseTensorShape(tensors_shape_map.at(name)[i % tensors_shape_map.at(name).size()]);
            } else if (info.partialShape.is_static()) {
                info.tensorShape = info.partialShape.get_shape();
            } else if (!tensors_shape_map.empty()) {
                throw std::logic_error("Wrong input names in tensor_shape command line parameter.");
            } else {
                throw std::logic_error(
                    "tensor_shape command line parameter should be set in case of network dynamic shape.");
            }

            // Layout
            if (layout_map.count(name)) {
                if (layout_map.at(name).size() > 1) {
                    throw std::logic_error(
                        "layout command line parameter doesn't support multiple layouts for one input.");
                }
                info._layout = descriptor.getLayout();
                info.layout = layout_map.at(name)[0];
                std::transform(info.layout.begin(), info.layout.end(), info.layout.begin(), ::toupper);
            } else {
                std::stringstream ss;
                ss << descriptor.getLayout();
                info._layout = descriptor.getLayout();
                info.layout = ss.str();
            }
            // Update shape with batch if needed (only in static case)
            // Update blob shape only not affecting network shape to trigger dynamic batch size case
            if (batch_size != 0) {
                std::size_t batch_index = info.layout.find("N");
                if ((batch_index != std::string::npos) && (info.tensorShape.at(batch_index) != batch_size)) {
                    if (info.partialShape.is_static()) {
                        info.partialShape[batch_index] = batch_size;
                    }
                    info.tensorShape[batch_index] = batch_size;
                    reshape_required = true;
                }
            }
            info_map[name] = info;
        }

        // Update scale and mean
        std::map<std::string, std::vector<float>> scale_map = parseScaleOrMean(scale_string, info_map);
        std::map<std::string, std::vector<float>> mean_map = parseScaleOrMean(mean_string, info_map);

        for (auto& item : info_map) {
            if (item.second.isImage()) {
                item.second.scale.assign({1, 1, 1});
                item.second.mean.assign({0, 0, 0});

                if (scale_map.count(item.first)) {
                    item.second.scale = scale_map.at(item.first);
                }
                if (mean_map.count(item.first)) {
                    item.second.mean = mean_map.at(item.first);
                }
            }
        }

        info_maps.push_back(info_map);
    }

    return info_maps;
}

template <typename T>
std::vector<benchmark_app::InputsInfo> getInputsInfo(const std::string& shape_string,
                                                     const std::string& layout_string,
                                                     const size_t batch_size,
                                                     const std::string& tensors_shape_string,
                                                     const std::string& scale_string,
                                                     const std::string& mean_string,
                                                     const std::map<std::string, T>& input_info) {
    bool reshape_required = false;
    return getInputsInfo<T>(shape_string,
                            layout_string,
                            batch_size,
                            tensors_shape_string,
                            scale_string,
                            mean_string,
                            input_info,
                            reshape_required);
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename, const std::map<std::string, std::map<std::string, std::string>>& config);
void load_config(const std::string& filename, std::map<std::string, std::map<std::string, std::string>>& config);
#endif
