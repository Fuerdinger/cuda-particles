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

#include "Image.h"

ImageManager::ImageManager()
{

}

ImageManager::ImageManager(const std::vector<Image*>& images)
{
	for (auto iter = images.begin(); iter != images.end(); iter++)
	{
		addImage(*iter);
	}
}

ImageManager::~ImageManager()
{
	for (auto iter = m_images.begin(); iter != m_images.end(); iter++)
	{
		delete iter->second;
	}
	m_images.clear();
}

void ImageManager::addImage(Image* image)
{
	auto iter = m_images.find(image->getName());
	assert(iter == m_images.end() || iter->second == nullptr);
	m_images[image->getName()] = image;
}

Image* ImageManager::getImage(const std::string& name)
{
	auto iter = m_images.find(name);
	if (iter == m_images.end() || iter->second == nullptr)
	{
		return nullptr;
	}
	return iter->second;
}

void ImageManager::deleteImage(const std::string& name)
{
	Image* image = removeImage(name);
	assert(image != nullptr);
	delete image;
}

Image* ImageManager::removeImage(const std::string& name)
{
	auto iter = m_images.find(name);
	if (iter == m_images.end() || iter->second == nullptr)
	{
		return nullptr;
	}
	Image* ret = iter->second;
	m_images.erase(iter);
	return ret;
}


std::string Image::_defaultPath = "";

//helper function for resizing CPU buffer (and potentially allocating a new one)
void Image::resizeCPUBuffer(GLsizei newWidth, GLsizei newHeight)
{
	//if data is loaded on the CPU, but it isn't the correct size, then delete the CPU buffer
	bool isWrongResolution = onCPU() && !(m_width == newWidth && m_height == newHeight);
	if (isWrongResolution)
	{
		free(m_pixels);
	}

	//if the CPU buffer has been deleted (or wasn't alloc'd to begin with), alloc one.
	if (isWrongResolution || !onCPU())
	{
		m_width = newWidth;
		m_height = newHeight;
		m_pixels = (unsigned char*)malloc(size_t(m_width) * m_height * m_pixelSize);
	}
}
void Image::resizeGPUBuffer()
{
	//if data is loaded on the GPU, but it isn't the correct size, then delete the GPU buffer
	bool isWrongResolution = onGPU() && !(m_width == m_gpuWidth && m_height == m_gpuHeight);
	if (isWrongResolution)
	{
		glDeleteTextures(1, &m_textureHandle);
	}

	//if the GPU buffer has been deleted (or wasn't alloc'd to begin with), alloc one.
	if (isWrongResolution || !onGPU())
	{
		m_gpuWidth = m_width;
		m_gpuHeight = m_height;

		//generate new texture and bind it
		glGenTextures(1, &m_textureHandle);
		glBindTexture(GL_TEXTURE_2D, m_textureHandle);

		//load data in and set parameters
		glTexImage2D(GL_TEXTURE_2D, 0, m_internalFormat, m_width, m_height, 0, m_format, m_type, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, m_filter);
		if (m_filter == GL_LINEAR_MIPMAP_LINEAR)
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		}
		else
		{
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, m_filter);
		}
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, m_clamp);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, m_clamp);

		//unbinds the texture
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

//doesn't do anything other than init vars
Image::Image(const std::string& name, GLint filter, GLint clamp, GLint internalformat, GLenum format, GLenum type, unsigned char pixelSize)
	: m_name(name),
	m_pixels(nullptr),
	m_textureHandle(-1),
	m_width(0), m_height(0),
	m_gpuWidth(0), m_gpuHeight(0),
	m_filter(filter),
	m_clamp(clamp),
	m_internalFormat(internalformat), m_format(format),
	m_type(type),
	m_pixelSize(pixelSize),
	m_bindIndex(-1)
{

}

//copy constructor
Image::Image(Image* otherImage, const std::string& name)
	: m_name(name),
	m_pixels(nullptr),
	m_textureHandle(-1),
	m_width(otherImage->m_width), m_height(otherImage->m_height),
	m_gpuWidth(otherImage->m_gpuHeight), m_gpuHeight(otherImage->m_gpuHeight),
	m_filter(otherImage->m_filter),
	m_clamp(otherImage->m_clamp),
	m_internalFormat(otherImage->m_internalFormat), m_format(otherImage->m_format),
	m_type(otherImage->m_type),
	m_pixelSize(otherImage->m_pixelSize),
	m_bindIndex(-1)
{
	if (otherImage->onCPU())
	{
		m_pixels = (unsigned char*)malloc(size_t(m_width) * m_height * m_pixelSize);
		if (m_pixels == nullptr) abort();
		memcpy(m_pixels, otherImage->m_pixels, size_t(m_width) * m_height * m_pixelSize);
	}
	if (otherImage->onGPU())
	{
		unsigned char* temp = m_pixels;
		m_pixels = (unsigned char*)malloc(size_t(m_gpuWidth) * m_gpuHeight * m_pixelSize);
		if (m_pixels == nullptr) abort();
		glBindTexture(GL_TEXTURE_2D, otherImage->m_textureHandle);
		glGetTexImage(GL_TEXTURE_2D, 0, m_format, m_type, m_pixels);
		glBindTexture(GL_TEXTURE_2D, 0);
		loadToGPUFromCPU();
		free(m_pixels);
		m_pixels = temp;
	}
}

//frees any resources inside the Image which haven't been freed yet
Image::~Image()
{
	if (onCPU()) freeFromCPU();
	if (onGPU()) freeFromGPU();
}

void Image::setDefaultPath(const std::string& path)
{
	_defaultPath = path;
}

std::string Image::getName() const
{
	return m_name;
}

//simple helper functions for seeing if data is on CPU or GPU
bool Image::onCPU() const
{
	return m_pixels != nullptr;
}
bool Image::onGPU() const
{
	return m_textureHandle != -1;
}

//getters for the width property; can only be called if it's on the CPU
GLsizei Image::getCPUWidth() const
{
	assert(onCPU());
	return m_width;
}
GLsizei Image::getCPUHeight() const
{
	assert(onCPU());
	return m_height;
}

//loads texture data into CPU buffer from either file system
//or from the current framebuffer
void Image::loadToCPUFromDisk(const std::string& path)
{
	int width, height, channels;
	if (!onCPU()) free(m_pixels);
	std::string file;
	if (path == "")
	{
		file = _defaultPath + m_name + ".png";
	}
	else
	{
		file = path + m_name + ".png";
	}
	stbi_set_flip_vertically_on_load(true);
	m_pixels = stbi_load(file.c_str(), &width, &height, &channels, STBI_rgb_alpha);
	stbi_set_flip_vertically_on_load(false);

	assert(m_pixels != nullptr);

	m_width = width;
	m_height = height;
}
void Image::loadToCPUFromFramebuffer(GLsizei width, GLsizei height)
{
	//width and height passed in must be equal to the width/height of the framebuffer
	resizeCPUBuffer(width, height);
	glReadPixels(0, 0, m_width, m_height, m_format, m_type, m_pixels);
}

//allocs the buffer and sets all values to (0,0,0,0)
void Image::loadToCPUFromNothing(GLsizei width, GLsizei height, unsigned char* pixels)
{
	resizeCPUBuffer(width, height);

	if (pixels == nullptr)
	{
		memset(m_pixels, 0, size_t(m_width) * m_height * m_pixelSize);
	}
	else
	{
		memcpy(m_pixels, pixels, size_t(m_width) * m_height * m_pixelSize);
	}
}

//copies data from gpu texture to cpu
void Image::loadToCPUFromGPU()
{
	assert(onGPU());
	resizeCPUBuffer(m_gpuWidth, m_gpuHeight);

	glBindTexture(GL_TEXTURE_2D, m_textureHandle);
	glGetTexImage(GL_TEXTURE_2D, 0, m_format, m_type, m_pixels);
	glBindTexture(GL_TEXTURE_2D, 0);
}

//this technically breaks our abstraction... but will make pixel-writing take less code.
unsigned char* Image::getCPUBuffer()
{
	assert(onCPU());
	return m_pixels;
}

//loads texture data from CPU buffer into a proper OpenGL texture
void Image::loadToGPUFromCPU()
{
	assert(onCPU());

	resizeGPUBuffer();

	//load pixels into texture
	glBindTexture(GL_TEXTURE_2D, m_textureHandle);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, m_format, m_type, m_pixels);
	if (m_filter == GL_LINEAR_MIPMAP_LINEAR)
	{
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	glBindTexture(GL_TEXTURE_2D, 0);

	//maybe in the future, allow a range to be specified, so the entire m_pixels isn't reloaded into GPU memory,
	//if only a small number of pixels actually changed
}

//creates the texture on the GPU without anything in it
void Image::loadToGPUFromNothing(GLsizei width, GLsizei height)
{
	m_width = width;
	m_height = height;

	resizeGPUBuffer();
}

//this will likely be replaced later to completely abstract the opengl texture from client
GLuint Image::getGPUHandle()
{
	assert(onGPU());
	return m_textureHandle;
}

//frees the CPU buffer (if not already freed)
void Image::freeFromCPU()
{
	assert(onCPU());
	free(m_pixels);
	m_pixels = nullptr;
}

//deletes the OpenGL texture (if it exists)
void Image::freeFromGPU()
{
	assert(onGPU());
	glDeleteTextures(1, &m_textureHandle);
	m_textureHandle = -1;
}

//if on the GPU, will bind/unbind the texture to a uniform sampler2D (or image2D)
void Image::bindToShader(unsigned int bindIndex)
{
	//be careful with this call.
	//changing the active texture can lead to strange behavior
	//example: calling bindToShader() on one Image, and then calling loadToGPUFromCPU() on another Image

	//note that whoever created the program is responsible for doing the uniform sampler binding.
	//glProgramUniform1i(program, glGetUniformLocation(program, ("myTextures[" + std::to_string(i) + "]").c_str()), i);
	
	assert(onGPU());
	glActiveTexture(GL_TEXTURE0 + bindIndex);
	glBindTexture(GL_TEXTURE_2D, m_textureHandle);
	m_bindIndex = bindIndex;
}
void Image::bindToShaderAsImage(GLenum access, unsigned int bindIndex)
{
	assert(onGPU());
	//g(glActiveTexture(GL_TEXTURE0 + bindIndex));
	glBindImageTexture(bindIndex, m_textureHandle, 0, GL_FALSE, 0, access, m_internalFormat);
	m_bindIndex = bindIndex;
}

//if on the GPU, will bind/unbind the texture to a uniform sampler2D (or image2D)
void Image::unbindToShader()
{
	//if binding and unbinding multiple textures, it is a good idea to unbind in reverse order.

	assert(onGPU());
	glActiveTexture(GL_TEXTURE0 + m_bindIndex);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	m_bindIndex = -1;
}
void Image::unbindToShaderAsImage()
{
	assert(onGPU());
	//the purpose of this is to unbind; here, GL_WRITE_ONLY and m_internalFormat are only passed to
	//avoid throwing an OpenGL error
	glBindImageTexture(m_bindIndex, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, m_internalFormat);
	m_bindIndex = -1;
}

//saves the image to disk (using the name as a filename)
void Image::saveToDiskFromCPU(const std::string& path)
{
	assert(onCPU());
	//currently, saving RGBA images with 4 bytes per pixel is supported
	assert(m_internalFormat == GL_RGBA8);

	//special chars must be removed so the file can be saved
	const char badChars[] = { '\\', '/', ':', '*', '?', '\"', '<', '>', '|' };
	std::string name = "";

	//for each character in the name
	for (size_t i = 0; i < m_name.length(); i++)
	{
		//determine whether the char should be added
		bool addChar = true;
		for (size_t j = 0; j < 9; j++)
		{
			if (m_name[i] == badChars[j])
			{
				addChar = false;
				break;
			}
		}

		//add it, if it isn't a special char
		if (addChar == true)
		{
			name += m_name[i];
		}
	}

	//erase characters from the back, if they are a period or whitespace
	while (name.length() > 0 && (name[name.length() - 1] == '.' || name[name.length() - 1] == ' '))
	{
		name.erase(name.length() - 1);
	}

	//if the name ended up being empty, just change name to "Image"
	if (name == "")
	{
		name = "Image";
	}

	std::string file;
	if (path == "")
	{
		file = _defaultPath + name + ".png";
	}
	else
	{
		file = path + name + ".png";
	}

	stbi_flip_vertically_on_write(true);
	stbi_write_png(file.c_str(), m_width, m_height, 4, m_pixels, 0);
	stbi_flip_vertically_on_write(false);
}
