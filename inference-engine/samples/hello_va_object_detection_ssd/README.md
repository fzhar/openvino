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

### Example
1. Download a pre-trained model using [Model Downloader](@ref omz_tools_downloader_README):
```
python <path_to_omz_tools>/downloader.py --name ssd_mobilenet_v1_coco
```

2. If a model is not in the Inference Engine IR or ONNX format, it must be converted. You can do this using the model converter script:

```
python <path_to_omz_tools>/converter.py --name ssd_mobilenet_v1_coco
```

3. Use the following command to do inference on GPU of a video using a trained SSD network:

```sh
hello_va_object_detection_ssd <path_to_model> <path_to_video>
```

with path to video file and [ssd_mobilenet_v1_coco](https://docs.openvinotoolkit.org/latest/omz_models_model_ssd_mobilenet_v1_coco.html) model

```sh
hello_va_object_detection_ssd <path_to_model>/ssd_mobilenet_v1_coco.xml <path_to_video>/inputVideo.mp4
```

## Sample Output

The application writes MJPG video file with detected objects enclosed in rectangles. It outputs the list of classes of the detected objects along with the respective confidence values and the coordinates of the rectangles to the standard output stream.

```sh
hello_va_object_detection_ssd person-detection-retail-0013/FP16/ssd_mobilenet_v1_coco.xml inputVideo.mp4

Resulting input shape = [1,3,300,300]
Resulting output shape = [1,1,100,7]
VA-API version 1.11.0

User environment variable requested driver 'iHD'

Trying to open /opt/intel/mediasdk/lib64/iHD_drv_video.so

va_openDriver() returns 0

Going to link Demux and Parser
Processing video
23 [0,19] element, prob = 0.52002, bbox = (893.534,155.55)-(1313.18,800)
24 [0,19] element, prob = 0.667969, bbox = (900.63,168.958)-(1297.15,800)
25 [0,19] element, prob = 0.720703, bbox = (892.454,159.218)-(1260.88,800)
26 [1,27] element, prob = 0.675293, bbox = (950.651,198.368)-(1236.63,428.756)
27 [0,1] element, prob = 0.539062, bbox = (940.214,146.898)-(1239.83,782.549)
28 [0,1] element, prob = 0.665527, bbox = (929.375,156.362)-(1227.07,800)
29 [0,1] element, prob = 0.600098, bbox = (897.009,162.408)-(1216.04,800)
30 [0,19] element, prob = 0.632324, bbox = (851.171,186.064)-(1219.76,788.946)
31 [0,19] element, prob = 0.544434, bbox = (737.024,189.676)-(1207.35,787.597)
32 [0,19] element, prob = 0.53125, bbox = (757.195,184.319)-(1227.7,786.459)
33 [0,1] element, prob = 0.59668, bbox = (898.113,170.464)-(1231.38,794.552)
36 [0,19] element, prob = 0.515137, bbox = (900.896,164.754)-(1238.69,800)
37 [0,1] element, prob = 0.602539, bbox = (930.521,197.939)-(1220.69,797.533)
38 [0,1] element, prob = 0.783691, bbox = (913.483,193.347)-(1222.67,791.42)
46 [0,1] element, prob = 0.520996, bbox = (907.94,346.572)-(1181.06,795.077)
48 [0,1] element, prob = 0.640625, bbox = (829.31,410.08)-(1287.35,790.787)
...
977 [0,1] element, prob = 0.523438, bbox = (198.33,271.002)-(650.132,760.953)
990 [0,1] element, prob = 0.524902, bbox = (625.645,427.941)-(823.217,727.706)
991 [0,1] element, prob = 0.525879, bbox = (329.455,351.062)-(607.585,792.528)
993 [0,1] element, prob = 0.520508, bbox = (538.617,348.751)-(837.473,751.732)
997 [0,1] element, prob = 0.59668, bbox = (539.594,368.884)-(846.195,754.577)
999 [0,1] element, prob = 0.586914, bbox = (524.67,373.395)-(841.19,767.772)
EOS received
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Remote Blob API of GPU Plugin](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_GPU_RemoteBlob_API.html)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
