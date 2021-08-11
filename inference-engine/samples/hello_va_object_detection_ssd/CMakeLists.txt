# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(va_video)

ie_add_sample(NAME hello_va_object_detection_ssd
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES ie_samples_utils va_video
              OPENCV_DEPENDENCIES core imgproc imgcodecs videoio)

find_package(ngraph REQUIRED)
target_link_libraries(hello_va_object_detection_ssd PRIVATE ${NGRAPH_LIBRARIES})
