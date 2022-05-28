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
#include "pch.h"

/**
 * @brief A static function wrapper class with useful utility functions.
 * @details A static pointer to an instance of this class is accessible
 * to Scene instances, allowing them access to the Util functions.
 * In the future, this class may be changed to have virtual functions, allowing
 * users to specify their own Util functions for usage by Scene instances, without having
 * to modify the Util class themselves.
 * 
 * @pre An OpenGL context must be initialized for the OpenGL calls to work.
 * 
 * @see Scene::_util
 * 
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class Util
{
private:
	static GLuint createProgramFromShaders(GLuint vertexShader, GLuint fragmentShader);
	static GLuint createComputeProgramFromShader(GLuint computeShader);

	static GLuint createShader(const std::string& code, GLenum type);
	static GLuint createShaderFromDisk(const std::string& location, GLenum type);
public:
	/**
	 * Creates a program of a vertex and fragment shader, or just a vertex shader.
	 * If compilation of either shader fails, or if the shaders can't be linked into a program, then
	 * an error message will be printed to stderr, and abort() will be called.
	 * @param[in] vertexShader The vertex shader's code. This is required, as all programs must have a vertex shader.
	 * @param[in] fragmentShader The fragment shader's code. This may be empty if no fragment shader is wanted.
	 * @return A handle to the newly created GLSL program.
	 */
	static GLuint createProgram(const std::string& vertexShader, const std::string& fragmentShader = "");

	/**
	 * Creates a program of a vertex and fragment shader, or just a vertex shader.
	 * If compilation of either shader fails, or if the shaders can't be linked into a program, then
	 * an error message will be printed to stderr, and abort() will be called.
	 * @param[in] vertexShaderPath Path to a file containing the vertex shader code. Should contain file extension.
	 * This is required, as all programs must have a vertex shader.
	 * @param[in] fragmentShaderPath Path to a file containing the fragment shader code. Should contain file extension.
	 * This may be empty if no fragment shader is wanted.
	 * @return A handle to the newly created GLSL program.
	 */
	static GLuint createProgramFromDisk(const std::string& vertexShaderPath, const std::string& fragmentShaderPath = "");

	/**
	 * Creates a program of a compute shader.
	 * If compilation fails, or the shader shader can't be linked into a program, then
	 * an error message will be printed to stderr, and abort() will be called.
	 * @param[in] computeShader The compute shader's code.
	 * @return A handle to the newly created GLSL program.
	 */
	static GLuint createComputeProgram(const std::string& computeShader);

	/**
	 * Creates a program of a compute shader.
	 * If compilation fails, or the shader shader can't be linked into a program, then
	 * an error message will be printed to stderr, and abort() will be called.
	 * @param[in] computeShaderPath Path to a file containing the compute shader code. Should contain file extension.
	 * @return A handle to the newly created GLSL program.
	 */
	static GLuint createComputeProgramFromDisk(const std::string& computeShaderPath);
};
