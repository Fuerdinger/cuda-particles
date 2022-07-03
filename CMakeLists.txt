#*****************************************************************************
# Copyright 2022 Daniel Fuerlinger
#
# This code is licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#  
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#****************************************************************************

cmake_minimum_required (VERSION 3.9)

# Sets up the toolchain for using vcpkg to download the packages in vcpkg.json
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "vcpkg toolchain file")
endif()

# vcpkg triplet to specify the hostadd
# (ex. x64-windows, x64-osx, etc)
if(DEFINED ENV{VCPKG_DEFAULT_HOST_TRIPLET} AND NOT DEFINED VCPKG_HOST_TRIPLET)
	set(VCPKG_HOST_TRIPLET "$ENV{VCPKG_DEFAULT_HOST_TRIPLET}" CACHE STRING "vcpkg host triplet")
endif()

# vcpkg triplet to specify the architecture, CRT linking, and whether to make a static/dynamic lib
# (ex. x64-windows-static, x64-windows-static-md)
if(DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
	set(VCPKG_TARGET_TRIPLET "$ENV{VCPKG_DEFAULT_TRIPLET}" CACHE STRING "vcpkg target triplet")
endif()

# We want to export an executable that is dynamically linked against the CRT
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "msvc runtime")

project("cuda-particles"
	VERSION 1.0.0
	DESCRIPTION "fork of cpp-opengl-game-framework for doing particle simulations"
	HOMEPAGE_URL "https://github.com/Fuerdinger/cpp-opengl-game-framework"
	LANGUAGES CXX CUDA
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${PROJECT_NAME}")

# The project currently doesn't use C++17 features but it may in the future
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# All these packages are available through vcpkg, and are specified in vcpkg.json
find_package(glad REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(SFML COMPONENTS audio REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_image.h" "stb_image_write.h")

# Specify source/header files and create the executable
# These are specified as vars to keep them separate from ${SOURCES} and ${INCLUDE_DIRECTORIES}
set(SRC_FILES
		"src/ext/stb_image.cpp"
		"src/ext/stb_image_write.cpp"
	"src/Image.cpp"
	"src/main.cpp"
	"src/Scene.cpp"
	"src/Sound.cpp"
	"src/Util.cpp"
	"src/Window.cpp"
)
set(INCLUDE_DIRS
	"include"
)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_precompile_headers(${PROJECT_NAME} PRIVATE "include/pch.h")

# Windows release builds should not be console apps
set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE $<CONFIG:Release>)

# Linking / including the libraries
target_link_libraries(${PROJECT_NAME} PRIVATE glad::glad)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)
target_link_libraries(${PROJECT_NAME} PRIVATE sfml-audio)
target_include_directories(${PROJECT_NAME} PRIVATE ${STB_INCLUDE_DIRS})

# Use doxygen to generate the documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)
	set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/docs")
	set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_LIST_DIR}/README.md")
	doxygen_add_docs(doxy
		${INCLUDE_DIRS}
		${SRC_FILES}
		"${CMAKE_CURRENT_LIST_DIR}/README.md"
		ALL
	)
endif()

# Copy asset data to the executable directory
# Deleting a file in root/data/ and then building again will NOT delete it from build/data/
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/data/")
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
		"${CMAKE_COMMAND}" -E copy_directory
		"${CMAKE_CURRENT_LIST_DIR}/data/"
		"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
	)
endif()
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/untracked_data/")
	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
		"${CMAKE_COMMAND}" -E copy_directory
		"${CMAKE_CURRENT_LIST_DIR}/untracked_data/"
		"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
	)
endif()

# Copy runtimes from System32 folder for Release builds
if(${CMAKE_BUILD_TYPE} MATCHES "Rel")
	if(DEFINED ENV{WINDIR})
		add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND
			"${CMAKE_COMMAND}" -E copy_if_different
			"$ENV{WINDIR}/System32/VCRUNTIME140.dll"
			"$ENV{WINDIR}/System32/VCRUNTIME140_1.dll"
			"$ENV{WINDIR}/System32/MSVCP140.dll"
			"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
		)
	endif()
endif()
