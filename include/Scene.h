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
#include "Window.h"
#include "Image.h"
#include "Sound.h"
#include "Util.h"

class Scene;

/**
 * @brief A class that manages Scene pointers using their names as keys.
 * @details A static pointer to an instance of this class is accessible
 * to Scene instances, allowing them to switch control flow to another Scene
 * or cause the game to exit. This class is the highest level class in the project,
 * and it is responsible for both initializing the game and running the main loop. 
 * 
 * @pre Only one SceneManager should be instantiated in the application.
 *
 * @see Scene
 * @see WindowManager::WindowManager(WindowManager::State, WindowManager::Mode)
 * @see ImageManager::ImageManager(const std::vector<Image*>& images)
 * @see SoundManager::SoundManager(const std::vector<std::string>&, const std::vector<std::string>&)
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class SceneManager
{
private:
	WindowManager* m_window;
	ImageManager* m_images;
	SoundManager* m_sounds;
	Util* m_util;

	std::unordered_map<std::string, Scene*> m_scenes;
	std::string m_currentSceneName;

	bool m_hasExited;

	void init(const std::vector<Scene*>& scenes, const std::string& firstScene);
	void update(std::chrono::steady_clock::time_point& timestamp);

public:
	/**
	 * Constructs a SceneManager. If the default argument for window is used,
	 * then this will construct its own WindowManager, which will create the game window
	 * and the OpenGL context.
	 * @param[in] window Window which a user may initialize themselves to give to the SceneManager.
	 * The SceneManager will become responsible for deleting this at deconstruction time.
	 */
	SceneManager(WindowManager* window = nullptr);

	/**
	 * Deconstructs the SceneManager. This will clean up all internal resources, including
	 * its WindowManager, ImageManager, SoundManager, and Util class instance. It will
	 * additionally delete all of its Scene instances.
	 */
	~SceneManager();

	/**
	 * Allows a user to inject their own ImageManager into the SceneManager.
	 * This is useful as it allows the user to load Image instances into an ImageManager and
	 * then pass it in here, so that Scene instances will have access to the Image instances at
	 * construction time.
	 * The SceneManager will become responsible for deleting this at deconstruction time.
	 * If the user doesn't do this themselves, then SceneManager will create its own ImageManager
	 * when build() is called.
	 * @pre This must be called before build().
	 */
	void createImages(ImageManager* images);

	/**
	 * Allows a user to inject their own SoundManager into the SceneManager.
	 * This is useful as it allows the user to load audio into an SoundManager and
	 * then pass it in here, so that Scene instances will have access to the audio at construction time.
	 * The SceneManager will become responsible for deleting this at deconstruction time.
	 * If the user doesn't do this themselves, then SceneManager will create its own SoundManager
	 * when build() is called.
	 * @pre This must be called before build().
	 */
	void createSounds(SoundManager* sounds);

	/**
	 * Gets the SceneManager ready to begin running the game loop.
	 * This is done by ensuring that all required manager classes are initialized,
	 * and then setting each of the static members of Scene to point to these instances.
	 * @post Scene constructors are now ensured that their static manager pointers will
	 * point to valid manager instances.
	 */
	void build();

	/**
	 * Runs the main game loop using the passed in Scene instances.
	 * The SceneManager will become responsible for deleting these Scene instances at
	 * deconstruction time.
	 * In this loop, a Scene will have its Scene::update(float) and Scene::render() functions
	 * called on each iteration.
	 * @param[in] scenes The Scene instances which the game should run using.
	 * @param[in] firstScene The name of the Scene which should be set to the current scene.
	 * If it is empty, then the first Scene in scenes will be set to the current scene.
	 * @pre This must be called after build().
	 * @pre scenes must have at least 1 Scene in it.
	 */
	void run(const std::vector<Scene*>& scenes, const std::string& firstScene = "");

	/**
	 * Changes the current active Scene to a new Scene, and allows it to pass data to the newly active Scene.
	 * @param[in] sceneName The name of the Scene to switch to.
	 * @param[in] data The data which the current active Scene may optionally pass to the newly active Scene.
	 * @pre This must be called by a Scene which is running inside run(const std::vector<Scene*>&, const std::string&)
	 * @pre sceneName must be the name of a valid Scene which was passed to run(const std::vector<Scene*>&, const std::string&)
	 */
	void switchTo(const std::string& sceneName, void* data = nullptr);

	/**
	 * Causes the game loop inside run(const std::vector<Scene*>&, const std::string&) to stop.
	 * @pre This must be called by a Scene which is running inside run(const std::vector<Scene*>&, const std::string&)
	 */
	void exit();
};

/**
 * @brief An inheritable class which encapsulates game behavior for an independent "scene."
 * @details A Scene instance, when run by a SceneManager, will be updated and rendered
 * on each frame. A user may inherit from this class to override the update and render
 * functions to define its behavior. Scene instances may use the Scene's static pointers to various
 * manager classes, which are set by the SceneManager, to access systems such as user input,
 * an image library, and a sound library.
 * 
 * @pre An OpenGL context must be initialized for the OpenGL calls to work.
 * @pre Scenes should not instantiated until SceneManager calls SceneManager::build()
 *
 * @see SceneManager::run(const std::vector<Scene*>&, const std::string&);
 * @see WindowManager
 * @see ImageManager
 * @see SoundManager
 * @see Util
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class Scene
{
public:
	/**
	 * @brief The input mode the Scene may be in.
	 */
	enum class Input_Mode
	{ 
		/// The mouse moves normally across the window for UI controls.
		FPP, 
		/// The mouse is grabbed by the window and is invisible for camera controls.
		UI 
	};

private:
	const std::string m_name;
	const Input_Mode m_inputMode;

protected:
	/// Pointer to instance of SceneManager class. Scene instances may use this to cause a Scene switch or make the game exit.
	static SceneManager* _scenes;
	/// Pointer to an instance of WindowManager class. Scene instances may use this to get info about the window, as well as user input.
	static WindowManager* _window;
	/// Pointer to an instance of ImageManager class. Scene instances may use this to access a shared library of OpenGL images with other Scene instances.
	static ImageManager* _images;
	/// Pointer to an instance of SoundManager class. Scene instances may use this to access a shared library of SFX and music with other Scene instances.
	static SoundManager* _sounds;
	/// Pointer to an instance of Util class. Scene instances may use its functions any way they like.
	static Util* _util;

public:
	/**
	 * Constructor which initializes a Scene.
	 * Once SceneManager::build() has been called, all static member variables of Scene will become valid pointers
	 * to instances of the various manager classes. This means that the Scene constructor will be able to safely
	 * dereference these pointers, and perform operations such as loading SFX and music and accessing images.
	 * Other things which Scene instances may do in their constructors include initializing OpenGL shaders or vertex
	 * buffers.
	 * @param[in] name The name of the Scene, which is how other Scene instances will refer to this one
	 * @param[in] inputMode Whether the Scene should use UI controls or first person camera controls
	 * @pre It is strongly suggested that no Scene instances be initialized until SceneManager::build() has been called.
	 */
	Scene(const std::string& name, Input_Mode inputMode = Input_Mode::UI);

	/**
	 * Destructor which destroys a Scene.
	 * All Scene instances passed to SceneManager::run(const std::vector<Scene*>&, const std::string&) will be deleted
	 * by the SceneManager in its destructor.
	 */
	virtual ~Scene();

	/**
	 * Gets the name of the Scene.
	 * @return The name of the Scene.
	 */
	std::string getName() const;

	/**
	 * Gets the Input_Mode of the Scene.
	 * @return The type of controls the Scene uses.
	 */
	Input_Mode getInputMode() const;

	/**
	 * Sets the static pointer to the SceneManager instance.
	 * @param[in] scenes A pointer to the SceneManager instance.
	 * @pre This is only supposed to be called by the SceneManager in
	 * SceneManager::build()
	 */
	static void setSceneManager(SceneManager* scenes);

	/**
	 * Sets the static pointer to the WindowManager instance.
	 * @param[in] window A pointer to the WindowManager instance.
	 * @pre This is only supposed to be called by the SceneManager in
	 * SceneManager::build()
	 */
	static void setWindowManager(WindowManager* window);

	/**
	 * Sets the static pointer to the ImageManager instance.
	 * @param[in] images A pointer to the ImageManager instance.
	 * @pre This is only supposed to be called by the SceneManager in
	 * SceneManager::build()
	 */
	static void setImageManager(ImageManager* images);

	/**
	 * Sets the static pointer to the SoundManager instance.
	 * @param[in] sounds A pointer to the SoundManager instance.
	 * @pre This is only supposed to be called by the SceneManager in
	 * SceneManager::build()
	 */
	static void setSoundManager(SoundManager* sounds);

	/**
	 * Sets the static pointer to the Util instance.
	 * @param[in] util A pointer to the Util instance.
	 * @pre This is only supposed to be called by the SceneManager in
	 * SceneManager::build()
	 */
	static void setUtil(Util* util);

	/**
	 * Called by SceneManager::run(const std::vector<Scene*>&, const std::string&) to "update"
	 * the Scene. A Scene instance may override this function to do anything it likes, but it
	 * generally should not attempt to render anything to the screen. Ideas on how to override this
	 * function include updating the internal state of the Scene based on input retrieved from
	 * _window, playing a sound effect retrieved from _sounds, updating a vertex buffer, etc.
	 * @param[in] deltaTime How much time has passed since the last call to update(float) was made.
	 */
	virtual void update(float deltaTime);

	/**
	 * Called by SceneManager::render() to "render" the Scene. A Scene instance should
	 * override this function to take its updated internal state and render it to the screen
	 * using OpenGL calls.
	 */
	virtual void render();

	/**
	 * Called by SceneManager::switchTo(const std::string&, void* data) when another
	 * Scene instance switches control flow to this Scene. Scene instances may override this function
	 * to perform any kind of behavior required for allowing the Scene to begin updating and rendering
	 * properly, such as resetting Scene state or using the passed in data to do something special.
	 * @param[in] previousScene The name of the Scene which switched control flow to this Scene.
	 * @param[in] data A pointer to any kind of data which the previous Scene may be handing to this Scene.
	 */
	virtual void switchFrom(const std::string& previousScene, void* data = nullptr);
};
