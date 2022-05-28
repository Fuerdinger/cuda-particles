/*****************************************************************************
Copyright 2022 Daniel Fuerlinger

This code is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*****************************************************************************/

#pragma once

//reduce warning level for 3rd party libraries
#pragma warning (push, 0)

//include windows debugging tools
#ifdef _WIN32
#include <Windows.h>
	#ifdef _DEBUG
#include <crtdbg.h>
#define _CRTDBG_MAP_ALLOC
	#endif
#endif

//audio system
#include <SFML/Audio.hpp>

//font rendering
#include <ft2build.h>
#include FT_FREETYPE_H

//GLFW window system, and GLAD for loading OpenGL
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

//stb_image for reading/writing PNG images
#include <stb_image.h>
#include <stb_image_write.h>

//GLM math library
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat4x2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <glm/gtx/quaternion.hpp>

//C runtime and STL
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <random>
#include <fstream>

//set warning level back to normal
#pragma warning (pop)
