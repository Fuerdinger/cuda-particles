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

#include "Util.h"

GLuint Util::createProgramFromShaders(GLuint vertexShader, GLuint fragmentShader)
{
	//create program and attach above shaders to link it
	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	if (fragmentShader != -1) glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	//make sure it linked successfully
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint logSize;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
		char* str = (char*)malloc(logSize);
		glGetProgramInfoLog(program, logSize, &logSize, str);
		fprintf(stderr, "Shader linkage error\nMessage: %s\n\n", str);
		free(str);
		abort();
	}

	//delete CPU resources and use the program
	glDeleteShader(vertexShader);
	if (fragmentShader != -1) glDeleteShader(fragmentShader);

	return program;
}

GLuint Util::createComputeProgramFromShader(GLuint computeShader)
{
	//create program and attach above shaders to link it
	GLuint program = glCreateProgram();
	glAttachShader(program, computeShader);
	glLinkProgram(program);

	//make sure it linked successfully
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status != GL_TRUE)
	{
		GLint logSize;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
		char* str = (char*)malloc(logSize);
		glGetProgramInfoLog(program, logSize, &logSize, str);
		fprintf(stderr, "Shader linkage error\nMessage: %s\n\n", str);
		free(str);
		abort();
	}

	//delete CPU resources and use the program
	glDeleteShader(computeShader);

	return program;
}

GLuint Util::createShader(const std::string& code, GLenum type)
{
	if (code == "") return -1;

	//create and compile shader
	GLuint shader = glCreateShader(type);
	const char* rawCode = code.c_str();
	glShaderSource(shader, 1, &rawCode, NULL);
	glCompileShader(shader);

	//make sure that it compiled successfully
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);

	if (status != GL_TRUE)
	{
		GLint logSize;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
		char* str = (char*)malloc(logSize);
		glGetShaderInfoLog(shader, logSize, &logSize, str);
		fprintf(stderr, "Shader compilation error\nMessage: %s\n\n", str);
		free(str);
		abort();
	}

	return shader;
}

GLuint Util::createShaderFromDisk(const std::string& location, GLenum type)
{
	if (location == "") return -1;

	//read string from specified file
	std::ifstream myStream(location);
	std::string myStr((std::istreambuf_iterator<char>(myStream)), (std::istreambuf_iterator<char>()));
	
	return createShader(myStr, type);
}

GLuint Util::createProgram(const std::string& vertexShader, const std::string& fragmentShader)
{
	//create vert and frag shaders
	GLuint vertShader = createShader(vertexShader, GL_VERTEX_SHADER);
	GLuint fragShader = createShader(fragmentShader, GL_FRAGMENT_SHADER);

	//create program from them
	return createProgramFromShaders(vertShader, fragShader);
}

GLuint Util::createProgramFromDisk(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
{
	//create vert and frag shaders
	GLuint vertShader = createShaderFromDisk(vertexShaderPath, GL_VERTEX_SHADER);
	GLuint fragShader = createShaderFromDisk(fragmentShaderPath, GL_FRAGMENT_SHADER);

	//create program from them
	return createProgramFromShaders(vertShader, fragShader);
}

GLuint Util::createComputeProgram(const std::string& computeShader)
{
	//create comp shader
	GLuint compShader = createShader(computeShader, GL_COMPUTE_SHADER);

	//create program from it
	return createComputeProgramFromShader(compShader);
}

GLuint Util::createComputeProgramFromDisk(const std::string& computeShaderPath)
{
	//create comp shader
	GLuint compShader = createShaderFromDisk(computeShaderPath, GL_COMPUTE_SHADER);

	//create program from it
	return createComputeProgramFromShader(compShader);
}
