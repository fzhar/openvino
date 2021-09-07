// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine.hpp>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <samples/ocv_common.hpp>
#include <string>
#include <vector>

#include <gpu/gpu_context_api_va.hpp>
#include <gpu/gpu_config.hpp>
#include "gst_vaapi_decoder.h"

using namespace InferenceEngine;

int main(int argc, char* argv[]) {
    const std::string output_filename = "out/hello_va_object_detection_output.avi";
    try {
        // ------------------------------ Parsing and validation of input arguments
        // ---------------------------------
        if (argc != 3) {
            std::cout << "Usage : " << argv[0] << " <path_to_model> <path_to_video>" << std::endl;
            return EXIT_FAILURE;
        }
        const std::string input_model {argv[1]};
        const std::string input_video_path {argv[2]};
        const std::string device_name {"GPU"};

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 1. Initialize inference engine core
        // -------------------------------------
        Core ie;

        // -----------------------------------------------------------------------------------------------------

        // Step 2. Read a model in OpenVINO Intermediate Representation (.xml and
        // .bin files) or ONNX (.onnx file) format
        CNNNetwork network = ie.ReadNetwork(input_model);

        OutputsDataMap outputs_info(network.getOutputsInfo());
        InputsDataMap inputs_info(network.getInputsInfo());
        if (inputs_info.size() != 1 || outputs_info.size() != 1)
            throw std::logic_error("Sample supports clean SSD network with one input and one output");

        // --------------------------- Setting batch size to 1
        auto input_shapes = network.getInputShapes();
        SizeVector& input_shape = input_shapes.begin()->second;
        input_shape[0] = 1;
        network.reshape(input_shapes);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 3. Configure input & output
        // ---------------------------------------------
        // --------------------------- Prepare input blobs
        // -----------------------------------------------------
        std::string input_name;
        InputInfo::Ptr input_info;
        std::tie(input_name, input_info) = *inputs_info.begin();
        // Set input layout and precision
        input_info->setLayout(Layout::NCHW);
        input_info->setPrecision(Precision::U8);
        // Setting input color format for automated conversion (NV12 provided to input will be automatically converted to model's color format)
        input_info->getPreProcess().setColorFormat(ColorFormat::NV12);

        // --------------------------- Prepare output blobs
        // ----------------------------------------------------
        DataPtr output_info;
        std::string output_name;
        std::tie(output_name, output_info) = *outputs_info.begin();
        // SSD has an additional post-processing DetectionOutput layer
        // that simplifies output filtering, try to find it.
        if (auto ngraphFunction = network.getFunction()) {
            for (const auto& op : ngraphFunction->get_ops()) {
                if (op->get_type_info() == ngraph::op::DetectionOutput::type_info) {
                    if (output_info->getName() != op->get_friendly_name()) {
                        throw std::logic_error("Detection output op does not produce a network output");
                    }
                    break;
                }
            }
        }

        const SizeVector output_shape = output_info->getTensorDesc().getDims();
        const size_t max_proposal_count = output_shape[2];
        const size_t object_size = output_shape[3];
        if (object_size != 7) {
            throw std::logic_error("Output item should have 7 as a last dimension");
        }
        if (output_shape.size() != 4) {
            throw std::logic_error("Incorrect output dimensions for SSD model");
        }
        if (output_info == nullptr) {
            IE_THROW() << "[SAMPLES] internal error - output information is empty";
        }

        output_info->setPrecision(Precision::FP32);

        auto dumpVec = [](const SizeVector& vec) -> std::string {
            if (vec.empty())
                return "[]";
            std::stringstream oss;
            oss << "[" << vec[0];
            for (size_t i = 1; i < vec.size(); i++)
                oss << "," << vec[i];
            oss << "]";
            return oss.str();
        };
        std::cout << "Resulting input shape = " << dumpVec(input_shape) << std::endl;
        std::cout << "Resulting output shape = " << dumpVec(output_shape) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 4. Load model
        // ------------------------------------------
        std::map<std::string, std::string> exec_network_config;
        // Setting number of simulatneously processing streams to 1 due to limitations of shared context operations
        exec_network_config.emplace(CONFIG_KEY(GPU_THROUGHPUT_STREAMS), "1");
        // Setting input color format for automated conversion (NV12 provided to input will be automatically converted to model's color format)
        exec_network_config.emplace(InferenceEngine::GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS,
            InferenceEngine::PluginConfigParams::YES);

        //----------------------------- Preparing Video Decoding and Processing objects ----------------
        GstVaApiDecoder decoder(input_shape[2],input_shape[3]);
        VaApiImage::Ptr decoded_frame;

        decoder.open(input_video_path);
        decoder.play();

        // --- Reading first frame
        bool keep_running = decoder.read(decoded_frame); // We have to read first frame in advance to get FPS and frame size

        auto& vaContext = decoded_frame->context;
        vaContext->createSharedContext(ie);
        auto sharedContext = vaContext->sharedContext();

        ExecutableNetwork executable_network = ie.LoadNetwork(network, sharedContext, exec_network_config);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 5. Create Infer request
        // -------------------------------------------------
        InferRequest infer_request = executable_network.CreateInferRequest();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- Step 6. Prepare input
        // --------------------------------------------------------

        
        cv::VideoWriter writer;
        if(!writer.open(output_filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),decoder.getFPS(),
            cv::Size(decoded_frame->width,decoded_frame->height))) {
            throw std::runtime_error("Cannot open "+output_filename);
        }

        // --- Main processing cycle
        std::cout<<"Processing video "<<std::endl;
        int frame_num = 0;
        while (keep_running) {
            
            // --- Resizing image to network's input size and putting it into blob
            auto resizedImg = decoded_frame;//->cloneToAnotherContext(vaContext);//->
//                resizeUsingPooledSurface(input_shape[3],input_shape[2], RESIZE_FILL,false);
            infer_request.SetBlob(input_name,
                InferenceEngine::gpu::make_shared_blob_nv12(input_shape[2], input_shape[3],
                sharedContext, resizedImg->va_surface_id));

            // --- Step 7. Infer
            infer_request.Infer();

            // --- Step 8. Process output
            cv::Mat image = decoded_frame->copyToMat();
            Blob::Ptr output = infer_request.GetBlob(output_name);
            MemoryBlob::CPtr moutput = as<MemoryBlob>(output);
            if (!moutput) {
                throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                    "but by fact we were not able to cast output to MemoryBlob");
            }
            // locked memory holder should be alive all time while access to its buffer
            // happens
            auto moutputHolder = moutput->rmap();
            const float* detection = moutputHolder.as<const float*>();

            // Each detection has image_id that denotes processed image
            for (size_t cur_proposal = 0; cur_proposal < max_proposal_count; cur_proposal++) {
                float image_id = detection[cur_proposal * object_size + 0];
                float label = detection[cur_proposal * object_size + 1];
                float confidence = detection[cur_proposal * object_size + 2];
                if (image_id < 0 || confidence == 0.0f) {
                    continue;
                }

                float xmin = detection[cur_proposal * object_size + 3] * image.cols;
                float ymin = detection[cur_proposal * object_size + 4] * image.rows;
                float xmax = detection[cur_proposal * object_size + 5] * image.cols;
                float ymax = detection[cur_proposal * object_size + 6] * image.rows;

                if (confidence > 0.5f) {
                    /** Drawing only objects with >50% probability **/
                    std::ostringstream conf;
                    conf << ":" << std::fixed << std::setprecision(3) << confidence;
                    cv::rectangle(image, cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax), cv::Scalar(0, 0, 255));
                    std::cout << frame_num << " [" << cur_proposal << "," << label << "] element, prob = " << confidence << ", bbox = (" << xmin << "," << ymin << ")-(" << xmax
                            << "," << ymax << ")"
                            << std::endl;
                }
            }

            // --- Writing frame to output video
            writer.write(image);
            frame_num++;

            // --- Reading next frame
            keep_running = decoder.read(decoded_frame);
        }

        std::cout << std::endl << "Processed " << frame_num << "frames."
                  << "The resulting image was saved in the file: "
                  << output_filename << std::endl;
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << std::endl
              << "This sample is an API example, for any performance measurements "
                 "please use the dedicated benchmark_app tool"
              << std::endl;
    return EXIT_SUCCESS;
}
