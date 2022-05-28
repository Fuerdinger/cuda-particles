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

#include "Scene.h"

void SceneManager::init(const std::vector<Scene*>& scenes, const std::string& firstScene)
{
	//put scenes in dict
	for (auto iter = scenes.begin(); iter != scenes.end(); iter++)
	{
		m_scenes[(*iter)->getName()] = *iter;
	}

	if (firstScene == "")
	{
		m_currentSceneName = scenes.front()->getName();
	}
	else
	{
		m_currentSceneName = firstScene;
	}

	switchTo(m_currentSceneName);
}

void SceneManager::update(std::chrono::steady_clock::time_point& timestamp)
{
	m_window->processPolls();

	std::string scene;

	//timekeeping
	auto currentTime = std::chrono::high_resolution_clock::now();
	float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - timestamp).count();

	//update the current scene; if a scene switch occurs, update the new current scene
	do
	{
		//ensure that the scene is actually in the list
		scene = m_currentSceneName;
		assert(m_scenes.find(m_currentSceneName) != m_scenes.end());

		//update the scene; the update function may change the SceneManager's internal state
		timestamp = std::chrono::high_resolution_clock::now();
		m_scenes[scene]->update(deltaTime);

		//if a scene switch occurred, consume the input to prevent
		if (scene != m_currentSceneName)
		{
			m_window->consumePolls();
		}

		//if a scene switch occurred, then m_currentSceneName is going to equal something else
	} while (scene != m_currentSceneName);

	//render whatever the current scene is now and swap the render buffer, assuming an exit hasn't occurred
	if (!m_hasExited)
	{
		m_scenes[m_currentSceneName]->render();
		m_window->swapBuffers();
	}
}

SceneManager::SceneManager(WindowManager* window)
{
	m_currentSceneName = "";	
	m_hasExited = false;

	m_window = window;
	if (m_window == nullptr) m_window = new WindowManager();
	
	m_images = nullptr;
	m_sounds = nullptr;

	m_util = new Util();
}
SceneManager::~SceneManager()
{
	//delete all the Scene pointers; their destructors will be called to free their resources
	for (auto iter = m_scenes.begin(); iter != m_scenes.end(); iter++)
	{
		delete iter->second;
	}
	m_scenes.clear();

	if (m_images != nullptr) delete m_images;
	if (m_sounds != nullptr) delete m_sounds;
	delete m_util;
	delete m_window;
}

void SceneManager::createImages(ImageManager* images)
{
	if (m_images != nullptr) delete m_images;
	m_images = images;
	Scene::setImageManager(m_images);
}

void SceneManager::createSounds(SoundManager* sounds)
{
	if (m_sounds != nullptr) delete m_sounds;
	m_sounds = sounds;
	Scene::setSoundManager(sounds);
}

void SceneManager::build()
{
	Scene::setSceneManager(this);
	Scene::setWindowManager(m_window);

	if (m_images == nullptr)
	{
		m_images = new ImageManager();
		Scene::setImageManager(m_images);
	}
	if (m_sounds == nullptr)
	{
		m_sounds = new SoundManager();
		Scene::setSoundManager(m_sounds);
	}

	Scene::setUtil(m_util);
}

//runs logic on scenes, performs potential scene switching, and renders
void SceneManager::run(const std::vector<Scene*>& scenes, const std::string& firstScene)
{
	init(scenes, firstScene);

	auto timestamp = std::chrono::high_resolution_clock::now();

	while (!(m_window->hasClosed() || m_hasExited))
	{
		update(timestamp);
	}
}

//changes the current scene to the new specified one
void SceneManager::switchTo(const std::string& sceneName, void* data)
{
	//ensure that the scene is actually in the list
	assert(m_scenes.find(sceneName) != m_scenes.end());

	//call the 'switch' event on the new scene
	m_scenes[sceneName]->switchFrom(m_currentSceneName, data);

	m_currentSceneName = sceneName;
	Scene::Input_Mode newInputMode = m_scenes[sceneName]->getInputMode();

	switch (newInputMode)
	{
	case Scene::Input_Mode::FPP:
		m_window->setMode(WindowManager::Mode::FPP);
		break;
	case Scene::Input_Mode::UI:
		m_window->setMode(WindowManager::Mode::UI);
		break;
	default:
		abort();
	}
}

//causes the game to exit the game loop in main
void SceneManager::exit()
{
	m_hasExited = true;
}

SceneManager* Scene::_scenes = nullptr;
WindowManager* Scene::_window = nullptr;
ImageManager* Scene::_images = nullptr;
SoundManager* Scene::_sounds = nullptr;
Util* Scene::_util = nullptr;

Scene::Scene(const std::string& name, Input_Mode inputMode)
	: m_name(name), m_inputMode(inputMode)
{

}

//virtual destructor for Scene
Scene::~Scene()
{

}

std::string Scene::getName() const
{
	return m_name;
}

Scene::Input_Mode Scene::getInputMode() const
{
	return m_inputMode;
}

void Scene::setSceneManager(SceneManager* scenes)
{
	_scenes = scenes;
}
void Scene::setWindowManager(WindowManager* window)
{
	_window = window;
}
void Scene::setImageManager(ImageManager* images)
{
	_images = images;
}
void Scene::setSoundManager(SoundManager* sounds)
{
	_sounds = sounds;
}
void Scene::setUtil(Util* util)
{
	_util = util;
}


void Scene::update(float deltaTime)
{

}
void Scene::render()
{

}
void Scene::switchFrom(const std::string& previousScene, void* data)
{

}
