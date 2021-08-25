# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (UNIX)
    find_package (PkgConfig)
    pkg_check_modules (GSTREAMER gstreamer-1.0 gstreamer-base-1.0 gstreamer-app-1.0 gstreamer-allocators-1.0 gstreamer-video-1.0)
    pkg_check_modules (VA libva)

    if(NOT GSTREAMER_FOUND)
	message(WARNING "gstreamer was not found, so hello_va_object_detection_ssd sample will not be built")
    elseif(NOT VA_FOUND)
	message(WARNING "libVA was not found, so hello_va_object_detection_ssd sample will not be built")
    else()
        add_subdirectory(va_video)

	ie_add_sample(NAME hello_va_object_detection_ssd
	          SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    		  DEPENDENCIES ie_samples_utils va_video
        	      OPENCV_DEPENDENCIES core imgproc imgcodecs videoio)

	find_package(ngraph REQUIRED)
        target_link_libraries(hello_va_object_detection_ssd PRIVATE ${NGRAPH_LIBRARIES})
    endif()
endif()
