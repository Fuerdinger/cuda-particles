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

class Image;

/**
 * @brief A class that manages Image pointers using their names as keys.
 * @details A static pointer to an instance of this class is accessible
 * to Scene instances, allowing the sharing of Images across Scenes.
 * This class may be explicitly initialized by a user of SceneManager
 * to allow for Image loading at startup.
 * 
 * @pre Only one ImageManager should be instantiated in the application.
 * 
 * @see Image
 * @see Scene::_images
 * @see SceneManager::createImages(ImageManager*)
 * 
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class ImageManager
{
private:
	std::map<std::string, Image*> m_images;

public:
	/**
	 * Constructs an ImageManager with no Image instances loaded in.
	 */
	ImageManager();

	/**
	 * Constructs an ImageManager with Image instances loaded in.
	 * These Image instances will become the property of the ImageManager,
	 * and should only be deleted or removed by the user using deleteImage(const std::string&)
	 * or removeImage(const std::string&). If the user does not call these functions on the Image
	 * instances of ImageManager, the ImageManager will clean up whatever Image instances are
	 * left in the destructor.
	 * @param[in] images The Image instances to load into the ImageManager.
	 * @pre images may not contain Image instances with the same names.
	 */
	ImageManager(const std::vector<Image*>& images);

	/**
	 * Deletes any Image instances left in the ImageManager.
	 */
	~ImageManager();

	/**
	 * Adds an Image instance to the ImageManager. This Image will become the property
	 * of the ImageManager, and should only be deleted or removed by the user using
	 * deleteImage(const std::string&) or removeImage(const std::string&). If the user
	 * does not call these functions, then the ImageManager will clean up the Image
	 * in its destructor.
	 * @param[in] image The Image instance to load in.
	 * @pre image may not have the same name as another Image already inside the ImageManager
	 */
	void addImage(Image* image);

	/**
	 * Gets an Image instance from the ImageManager by its name.
	 * @param[in] name The name of the Image to retrieve.
	 * @return The Image with a matching name, nullptr if no such Image exists.
	 */
	Image* getImage(const std::string& name);

	/**
	 * Removes an Image instance from the ImageManager by name, and then deletes it.
	 * @param[in] name The name of the Image to delete.
	 * @pre the name must associate with a valid Image in the ImageManager.
	 */
	void deleteImage(const std::string& name);

	/**
	 * Removes an Image instance from the ImageManager by name. The user
	 * will be responsible for deleting it.
	 * @param[in] name The name of the Image to remove.
	 * @return The Image with a matching name, nullptr if no such Image exists.
	 */
	Image* removeImage(const std::string& name);
};

/**
 * @brief A class that wraps around an OpenGL texture.
 * @details An Image may be loaded into CPU memory by a variety of means
 * (including PNG file, current framebuffer, or even "hand-written" in code)
 * then subsequently loaded to GPU memory for usage in a shader.
 * An Image may also be manipulated via directly editing its pixels
 * in CPU memory or using a compute shader to edit its GPU memory.
 * An Image may be saved to disk as a PNG.
 * 
 * @pre An OpenGL context must be initialized for the OpenGL calls to work.
 *
 * @see ImageManager
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class Image
{
private:
	const std::string m_name;

	static std::string _defaultPath;

	//contains the RGBA data for the image;
	//null when not loaded onto CPU
	unsigned char* m_pixels;

	//handle to GPU texture for image;
	//-1 when not loaded onto GPU
	GLuint m_textureHandle;

	//when bound, this is assigned to the bind index
	unsigned int m_bindIndex;

	GLsizei m_width;
	GLsizei m_height;

	//size of image on GPU; separate from CPU to check for when user
	//resizes the image, while the image is on the GPU
	GLsizei m_gpuWidth;
	GLsizei m_gpuHeight;

	//properties needed for putting image on GPU
	const GLint m_filter;
	const GLint m_clamp;
	const GLint m_internalFormat;
	const GLenum m_format;
	const GLenum m_type;

	//the size (in bytes) of a single pixel on the CPU
	const unsigned char m_pixelSize;

	//helper function for resizing CPU and GPU buffers (and potentially just creating a new one)
	void resizeCPUBuffer(GLsizei newWidth, GLsizei newHeight);
	void resizeGPUBuffer();

public:
	/**
	 * Constructs an Image, but doesn't do anything (ie no loading to CPU or GPU).
	 * The parameters passed in here will be used for future operations on the Image.
	 * @param[in] name The name of the Image.
	 * @param[in] filter If loaded onto the GPU, how the Image should be filtered.
	 * @param[in] clamp If loaded onto the GPU, how the Image should be clamped.
	 * @param[in] internalformat If loaded onto the GPU, what the internal format of the Image should be.
	 * @param[in] format If loaded onto the GPU, what the format of the Image should be.
	 * @param[in] type If loaded onto the GPU, what the pixel type should be.
	 * @param[in] pixelSize If loaded onto the CPU, how many bytes per pixel.
	 * @pre All GLint and GLenum arguments should be proper OpenGL values.
	 * @see https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexImage2D.xhtml
	 * @see https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glTexParameter.xhtml
	 */
	Image(const std::string& name, GLint filter = GL_NEAREST, GLint clamp = GL_REPEAT, GLint internalformat = GL_RGBA8, GLenum format = GL_RGBA, GLenum type = GL_UNSIGNED_BYTE, unsigned char pixelSize = 4);

	/**
	 * Copy constructor. Makes a deep copy, and if the other Image is on the CPU and GPU,
	 * its CPU and GPU state will be deep copied as well.
	 * @param[in] otherImage The Image to make a deep copy of.
	 * @param[in] name The name of the new Image to created.
	 */
	Image(Image* otherImage, const std::string& name);

	/**
	 * Destructor which frees any resources of the Image which have not been freed already
	 * in calls to freeFromCPU() or freeFromGPU().
	 */
	~Image();

	/**
	 * Sets the default directory where Image instances will be loaded from in calls
	 * to loadToCPUFromDisk(const std::string&). If not an empty string, should end with
	 * a slash.
	 * @param[in] path The directory where images reside on disk. Should end with a slash, unless
	 * it is empty
	 */
	static void setDefaultPath(const std::string& path);

	/**
	 * Gets the name of the Image
	 * @return The name of the Image
	 */
	std::string getName() const;

	/**
	 * Gets whether the Image currently has pixel data residing in CPU
	 * memory. This would be the case if any of the loadToCPUFromX functions
	 * are called, and freeFromCPU() has not been called.
	 * @return True if the Image is on the CPU, False if not
	 */
	bool onCPU() const;

	/**
	 * Gets whether the Image currently is loaded onto the GPU as an OpenGL texture.
	 * This would be the case if any of the loadToGPUFromX functions are called,
	 * and freeFromGPU() has not been called.
	 * @return True if the Image is on the GPU, False if not
	 */
	bool onGPU() const;

	//getters for the width property; can only be called if it's on the CPU

	/**
	 * Gets the width, in terms of pixels, of the CPU buffer of the Image.
	 * Note that this may be different from the pixel width of the Image if it
	 * is loaded onto the GPU, because the CPU state of an Image may change
	 * without changing its GPU state.
	 * @return The width of the CPU pixel buffer.
	 * @pre The Image must be on the CPU.
	 */
	GLsizei getCPUWidth() const;
	
	/**
	 * Gets the height, in terms of pixels, of the CPU buffer of the Image.
	 * Note that this may be different from the pixel height of the Image if it
	 * is loaded onto the GPU, because the CPU state of an Image may change
	 * without changing its GPU state.
	 * @return The height of the CPU pixel buffer.
	 * @pre The Image must be on the CPU.
	 */
	GLsizei getCPUHeight() const;

	/**
	 * Loads a PNG image into the CPU pixel buffer of this Image.
	 * This data may be manipulated directly via getCPUBuffer()
	 * and may be loaded to the GPU as an OpenGL texture via loadToGPUFromCPU().
	 * @param[in] path The directory where the PNG image resides. If left as empty,
	 * the default directory set via setDefaultPath(const std::string&) will be used.
	 * @pre A PNG must reside at path + name + .png
	 * @post If not already on the CPU, the Image will now be on the CPU
	 */
	void loadToCPUFromDisk(const std::string& path = "");

	/**
	 * Loads pixel data into the CPU pixel buffer of this Image from the currently bound
	 * framebuffer; by default, this framebuffer will belong to the window surface itself.
	 * @param[in] width The number of pixels wide that should be read from the framebuffer
	 * @param[in] height The number of pixels high that should be read from the framebuffer
	 * @pre width and height must be less than or equal to the size of the framebuffer
	 * @post If not already on the CPU, the Image will now be on the CPU
	 */
	void loadToCPUFromFramebuffer(GLsizei width, GLsizei height);

	//allocs the buffer and sets all values to (0,0,0,0)

	/**
	 * Loads pixel data into the CPU pixel buffer of this Image, as empty or from a preexisting
	 * pixel buffer.
	 * @param[in] width The number of pixels wide the pixel buffer should be.
	 * @param[in] height The number of pixels high the pixel buffer should be.
	 * @param[in] pixels If this is specified, then the pixel data will be copied from this. If
	 * not specified, then each pixel value will be set to 0.
	 * @pre If pixels is specified, then its binary size must be equal to width * height * the pixel
	 * size of this Image.
	 * @post If not already on the CPU, the Image will now be on the CPU
	 */
	void loadToCPUFromNothing(GLsizei width, GLsizei height, unsigned char* pixels = nullptr);

	/**
	 * Loads pixel data into the CPU pixel buffer of this Image from the pixel data stored
	 * on the OpenGL texture.
	 * @pre The Image must be on the GPU
	 */
	void loadToCPUFromGPU();

	/**
	 * Gets a pointer to the CPU pixel buffer of the Image. Its size is equal to width * height *
	 * the size of a pixel. This function is sort of dangerous as it breaks the abstraction which
	 * the Image represents, but it is provided for convenience over safety.
	 * @return A pointer to the CPU pixel buffer
	 * @pre The Image is on the CPU
	 */
	unsigned char* getCPUBuffer();

	/**
	 * Loads the data stored in the CPU pixel buffer of the Image into an OpenGL texture, which
	 * may be bound for usage in shaders. The parameters specified in the constructor will be
	 * used to determine how the Image will be sampled in a shader.
	 * @pre The Image must be on the CPU.
	 * @post If not already on the GPU, the Image will now be on the GPU in addition to the CPU
	 */
	void loadToGPUFromCPU();

	/**
	 * Loads empty pixel data into an OpenGL texture. The parameters specified in the constructor
	 * will be used to determine how the Image will be sampled in a shader.
	 * @param[in] width The number of pixels wide the OpenGL texture should be
	 * @param[in] height The number of pixels high the OpenGL texture should be
	 * @post If not already on the GPU, the Image will now be on the GPU
	 */
	void loadToGPUFromNothing(GLsizei width, GLsizei height);

	/**
	 * Obtains a handle to the OpenGL texture representation of the Image, which may be used
	 * in a variety of OpenGL calls such as binding the texture to a newly created framebuffer.
	 * This function is sort of dangerous as it breaks the abstraction which
	 * the Image represents, but it is provided for convenience over safety.
	 * @pre The Image must be on the GPU
	 */
	GLuint getGPUHandle();

	/**
	 * Frees the CPU pixel buffer of the Image.
	 * @pre The Image must be on the CPU.
	 * @post The Image will no longer be on the CPU.
	 */
	void freeFromCPU();

	/**
	 * Deletes the OpenGL texture representation of the Image.
	 * @pre The Image must be on the GPU.
	 * @post The Image will no longer be on the GPU.
	 */
	void freeFromGPU();

	/**
	 * Binds the OpenGL texture to a sampler2D slot in a shader program via a bind index.
	 * This function is very dangerous if used unwisely, because it changes the active
	 * OpenGL texture to be GL_TEXTURE0 + bindIndex. If another Image calls loadToGPUFromCPU()
	 * after this Image calls bindToShader(unsigned int), for example, then unexpected behavior
	 * will be likely occur.
	 * @param[in] bindIndex The slot of the uniform sampler2D in the shader which the Image should be bound to.
	 * @pre The Image is on the GPU
	 */
	void bindToShader(unsigned int bindIndex = 0);

	/**
	 * Binds the OpenGL texture to an image2D slot in a shader program via a bind index.
	 * @param[in] access The type of access (readonly, writeonly, etc) the image2D should have
	 * @param[in] bindIndex The slot of the uniform image2D in the shader which the Image should be bound to.
	 * @pre The Image is on the GPU
	 * @pre access is a valid GLenum for glBindImageTexture
	 * @see https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBindImageTexture.xhtml
	 */
	void bindToShaderAsImage(GLenum access, unsigned int bindIndex = 0);

	/**
	 * Unbinds the OpenGL texture from its previously bound index.
	 * If multiple Image instances were bound at the same time, then they should be unbound in
	 * the reverse order of the order they were bound in.
	 * @pre The Image is on the GPU and was previously bound using bindToShader(unsigned int)
	 */
	void unbindToShader();

	/**
	 * Unbinds the OpenGL texture from its previously bound index.
	 * @pre The Image is on the GPU and was previously bound using bindToShaderAsImage(GLenum, unsigned int)
	 */
	void unbindToShaderAsImage();

	/**
	 * Saves the pixel data in the CPU buffer of the Image as a PNG to disk. The PNG file
	 * will be path + name + .png. If a PNG file already exists at this location, it will
	 * be overridden. If name contains characters which are improper for a Windows file, they
	 * will be omitted. If the file edited name doesn't have any characters in it, then the name "Image"
	 * will be used instead.
	 * @param[in] path The directory where the PNG file should be saved to. Should end with a slash if not
	 * empty. If empty, then the default path set with setDefaultPath(const std::string&) will be used.
	 * @pre The Image must be on the CPU
	 */
	void saveToDiskFromCPU(const std::string& path = "");
};
