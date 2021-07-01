# Hello VA Object Detection SSD C++ Sample {#openvino_inference_engine_samples_hello_va_object_detection_ssd_README}

This sample demonstrates how to execute an inference of object detection networks like SSD-VGG for images stored in video memory using Synchronous Inference Request API using [Remote Blob API of GPU Plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API.html). Input images are decoded from video using [gstreamer](https://gstreamer.freedesktop.org/) and are kept in video memory. Remote blob API usage eliminates extra data copying from video decoder to Inference Engine and speeds up application.

Hello VA Object Detection SSD C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
| Network Operations | `InferenceEngine::CNNNetwork::getBatchSize`, `InferenceEngine::CNNNetwork::getFunction` |  Managing of network, operate with its batch size.
|Input Reshape|`InferenceEngine::CNNNetwork::getInputShapes`, `InferenceEngine::CNNNetwork::reshape`| Resize network to match image sizes and given batch
|nGraph Functions|`ngraph::Function::get_ops`, `ngraph::Node::get_friendly_name`, `ngraph::Node::get_type_info`| Go thru network nGraph
|Input Preprocessing|`InferenceEngine::InputInfo::getPreprocess`, `InferenceEngine::PreProcessInfo::setColorFormat`| Set input color format
|Blob operations|`InferenceEngine::gpu::make_shared_blob_nv12`|Create shared memory blob

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | Person detection SSD (object detection network)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin), ONNX (\*.onnx)
| Validated images                 | The sample uses gstreamer to read frames from video files (any video format supported by gstreamer)
| Supported devices                | [GPU](../../../docs/IE_DG/supported_plugins/GPU.md) |
| Supported OS       | Linux |

## How It Works

Upon the start-up the sample application reads command line parameters, loads specified network to the Inference Engine plugin. Then, sample decodes video frame using gstreamer, resizes it to the network input size and keeps in video memory (as VA surface). Then, sample creates an synchronous inference request object that shares video memory with decoded frame (to avoid data copying). When inference is done, the application writes frame to the output video file and outputs data to the standard output stream. Then sample decodes next frame and repeats all the following steps until the video is over.

NOTE: It is essential to avoid destruction/recreating of VA surfaces passed to Inference Engine plugin as memory handles are cached inside the plugin. Surfaces can be reused (i.e. new frame might be saved there) after inference is done, but they should not be destroyed and recreated as it will break caching. Because of that VA surfaces used to store results of resize (that will be passed to inference) are created only onces and stored inside the pool of surfaces.

You can see the explicit description of sample steps at [Integration Steps](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

To run the sample, you need specify a model and input video file:

- you can use [public](@ref omz_models_public_index) or [Intel's](@ref omz_models_intel_index) pre-trained models from the Open Model Zoo. The models can be downloaded using the [Model Downloader](@ref omz_tools_downloader_README).
- you can use videos   from the media files collection available at https://storage.openvinotoolkit.org/data/test_data.

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (\*.onnx) that do not require preprocessing.

You can use the following command to do inference on CPU of an image using a trained SSD network:

```sh
hello_va_object_detection_ssd <path_to_model> <path_to_video>
```

with path to video file and [person-detection-retail-0013](https://docs.openvinotoolkit.org/latest/omz_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html) model

```sh
hello_va_object_detection_ssd <path_to_model>/person-detection-retail-0013.xml <path_to_video>/inputVideo.mp4
```

## Sample Output

The application writes mjpg video file with detected objects enclosed in rectangles. It outputs the list of classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.

```sh
hello_va_object_detection_ssd person-detection-retail-0013/FP16/person-detection-retail-0013.xml inputVideo.mp4

Resulting input shape = [1,3,320,544]
Resulting output shape = [1,1,200,7]
VA-API version 1.11.0

User environment variable requested driver 'iHD'

Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so

va_openDriver() returns 0

Going to link Demux and Parser
Going to link Demux and Parser
Processing video 
0 [0,1] element, prob = 0.528809, bbox = (978.647,173.295)-(1225.8,780.653)
1 [0,1] element, prob = 0.566895, bbox = (1341.04,533.854)-(1444.96,800.08)
1 [1,1] element, prob = 0.541016, bbox = (757.027,549.43)-(869.127,800.135)
2 [0,1] element, prob = 0.708984, bbox = (1345.85,512.396)-(1449.06,803.336)
4 [0,1] element, prob = 0.82959, bbox = (1320.37,503.603)-(1431.64,795.653)
9 [0,1] element, prob = 0.94043, bbox = (884.536,565.753)-(995.223,797.401)
9 [1,1] element, prob = 0.535156, bbox = (760.536,558.299)-(878.711,795.652)
11 [0,1] element, prob = 0.650879, bbox = (890.801,537.985)-(1046.12,795.818)
18 [0,1] element, prob = 0.844727, bbox = (961.421,153.04)-(1121.84,459.374)
18 [1,1] element, prob = 0.539062, bbox = (934.218,622.477)-(1021.39,801.726)
19 [0,1] element, prob = 0.913574, bbox = (959.474,142.192)-(1117.74,447.63)
19 [1,1] element, prob = 0.667969, bbox = (1005.37,150.718)-(1187.58,508.281)
20 [0,1] element, prob = 0.853516, bbox = (957.201,136.085)-(1121.88,448.934)
20 [1,1] element, prob = 0.853516, bbox = (833.783,535.044)-(915.237,798.918)
20 [2,1] element, prob = 0.647949, bbox = (904.83,562.651)-(1001.86,799.35
...
997 [0,1] element, prob = 0.926758, bbox = (1282.26,330.531)-(1446.75,750.631)
997 [1,1] element, prob = 0.859863, bbox = (1068.61,437.713)-(1309.69,799.561)
997 [2,1] element, prob = 0.747559, bbox = (340.615,372.104)-(573.225,796.316)
998 [0,1] element, prob = 0.9375, bbox = (1286.3,332.543)-(1447.63,739.889)
998 [1,1] element, prob = 0.918457, bbox = (1069.63,439.769)-(1308.97,801.97)
998 [2,1] element, prob = 0.858887, bbox = (348.487,368.039)-(554.498,794.911)
999 [0,1] element, prob = 0.936035, bbox = (1287.76,326.534)-(1449.67,740.98)
999 [1,1] element, prob = 0.922363, bbox = (1059.18,440.228)-(1310.54,802.218)
999 [2,1] element, prob = 0.742676, bbox = (340.102,375.159)-(561.801,798.802)
EOS received
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Remote Blob API of GPU Plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API.html)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
