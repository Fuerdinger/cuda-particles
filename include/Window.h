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
 * @brief A class that wraps around a window. Currently uses GLFW.
 * @details A static pointer to an instance of this class is accessible
 * to Scene instances, allowing them access to obtaining user input and some
 * of the window's properties.
 * This class may be explicitly initialized by a user of SceneManager
 * to allow for non-standard window creation.
 * 
 * @pre Only one WindowManager may be instantiated in the application.
 * @post Instantiating a WindowManager will create an OpenGL context, which
 * the rest of the application will use.
 *
 * @see Scene::_window
 * @see SceneManager::SceneManager(WindowManager*)
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class WindowManager
{
public:
	/**
	 * @brief The states a key or mouse button input may take.
	 */
	enum class Action
	{ 
		/// Inactive.
		NONE,
		/// Went from not being held down to being pressed.
		PRESS,
		/// Went from being held down to being inactive.
		RELEASE,
		/// Pressed for more than one cycle.
		HOLD,
		/// Special state indicating that the input should not be taken into consideration.
		CONSUMED
	};
	/**
	 * @brief The states the window may be in.
	 */
	enum class State 
	{ 
		/// The window is a floating, decorated window which can be moved, minimized, maximized, and closed.
		WINDOWED, 
		/// The window is fullscreen.
		FULLSCREEN, 
		/// The window is borderless and is the resolution of the entire screen. This is essentially "fake" fullscreen.
		BORDERLESS
	};
	/**
	 * @brief The input mode the window may be in.
	 */
	enum class Mode
	{ 
		/// The mouse moves normally across the window for UI controls.
		UI, 
		/// The mouse is grabbed by the window and is invisible for camera controls.
		FPP 
	};

private:
	static const int defaultKeys[];
	static const int defualtMouseButtons[];

	//window state
	GLFWwindow* m_window;
	State m_state;
	Mode m_mode;
	bool m_minimized;
	bool m_focused;
	bool m_containsMouse;
	bool m_rawMotionSupported;

	//window dimensions
	glm::ivec2 m_framebufferSize;
	glm::ivec2 m_size;
	glm::ivec2 m_pos;
	glm::vec2 m_scale;

	//user input state
	std::map<int, Action> m_keyStates;
	std::map<int, Action> m_mouseStates;
	glm::ivec2 m_cursorPos;
	glm::ivec2 m_cursorDelta;

	//callbacks
	static void glfwCursorEnterCallback(GLFWwindow* window, int entered);
	static void glfwWindowCloseCallback(GLFWwindow* window);
	static void glfwWindowFocusCallback(GLFWwindow* window, int focused);
	static void glfwWindowIconifyCallback(GLFWwindow* window, int iconified);
	static void glfwFramebufferResizeCallback(GLFWwindow* window, int width, int height);
	static void glfwWindowSizeCallback(GLFWwindow* window, int width, int height);
	static void glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos);
	static void glfwWindowContentScaleCallback(GLFWwindow* window, float xscale, float yscale);
	void cursorEnterCallback(int entered);
	void windowCloseCallback();
	void windowFocusCallback(int focused);
	void windowIconifyCallback(int iconified);
	void framebufferResizeCallback(int width, int height);
	void windowSizeCallback(int width, int height);
	void windowPosCallback(int xpos, int ypos);
	void windowContentScaleCallback(float xscale, float yscale);

	Action assignAction(Action action1, Action action2);

	void switchState(State state);
	void switchMode(Mode mode);
public:
	/**
	 * Constructs a window with an OpenGL context.
	 * Also registers a default set of keyboard keys and mouse buttons to keep track of input for.
	 * These include WASD, the arrow keys, spacebar, enter, escape, LMB, RMB, and MMB
	 * @param[in] name The text which should be put into the window's decoration (if not fullscreen)
	 * @param[in] state How the window should be created to start with.
	 * @param[in] mode The input mode the window should have to start with.
	 * @pre Only one WindowManager may be instantiated in the application.
	 * @post Instantiating a WindowManager will create an OpenGL context, which the
	 * rest of the application will use.
	 */
	WindowManager(const std::string& name = "Game", State state = State::WINDOWED, Mode mode = Mode::UI);

	/**
	 * Destructs the window and the OpenGL context.
	 */
	~WindowManager();

	/**
	 * Updates the input state of the WindowManager for reading on the current frame.
	 */
	void processPolls();

	/**
	 * Sets any inputs with the Action::PRESS state to be Action::CONSUMED
	 * This can be used to prevent an Action::PRESS input from being read twice.
	 */
	void consumePolls();

	/**
	 * Swaps the front and back drawing buffers of the window.
	 */
	void swapBuffers();

	/**
	 * Sets the flag to close the window. The window will attempt
	 * to close when the flag is next read.
	 */
	void signalClose();

	/**
	 * Gets whether the window has actually closed or not.
	 * @return True if the window has closed, false if not
	 */
	bool hasClosed() const;

	/**
	 * Adds a keyboard key to keep track of input for.
	 * @param[in] key The key to register. Should be GLFW_KEY_X
	 * @pre The key hasn't already been registered yet.
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	void registerKey(int key);

	/**
	 * Adds a mouse button to keep track of input for.
	 * @param[in] button The mouse button to register. Should be GLFW_MOUSE_BUTTON_X
	 * @pre The mouse button hasn't already been registered yet.
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	void registerMouseButton(int button);

	/**
	 * Gets the current state of the window.
	 * @return the State of the window.
	 */
	State getState() const;

	/**
	 * Gets the current input mode of the window.
	 * @return the Mode of the window.
	 */
	Mode getMode() const;

	/**
	 * Gets the size of the window's framebuffer. Not necessarily
	 * equal to the size of the window itself, making this the more
	 * important value for use with OpenGL calls.
	 * @return The size of the window's framebuffer.
	 */
	glm::ivec2 getFramebufferSize() const;

	/**
	 * Gets the size of the window. Not necessarily equal to the size
	 * of the framebuffer, making this value somewhat irrelevant.
	 * @return The size of the window.
	 */
	glm::ivec2 getSize() const;

	/**
	 * Gets the position of the window in the user's monitor workspace.
	 * @return The position of the window.
	 */
	glm::ivec2 getPos() const;

	/**
	 * Gets the content scale of the window. A scale of 100% will be 1.0,
	 * and a scale of 250% will be 2.5
	 * @return The scale of the window.
	 */
	glm::vec2 getScale() const;

	/**
	 * Sets the State of the window.
	 * @param[in] state The new State of the window.
	 */
	void setState(State state);

	/**
	 * Sets the input Mode of the window.
	 * @param[in] mode The new input Mode of the window.
	 */
	void setMode(Mode mode);

	/**
	 * Sets the size which the window should occupy in the user's monitor workspace.
	 * @param[in] windowSize The new size of the window.
	 */
	void setSize(const glm::ivec2& windowSize);

	/**
	 * Sets the position which the window should occupy in the user's monitor workspace.
	 * @param[in] windowPos The new position of the window.
	 */
	void setPos(const glm::ivec2& windowPos);

	/**
	 * Gets the Action state of the desired keyboard key.
	 * @param[in] key The key to read the input of. Should be GLFW_KEY_X
	 * @pre key should be a valid registered key.
	 * @return the Action state of the desired key.
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	Action getKey(int key) const;

	/**
	 * Checks if the Action state of a keyboard key is Action::PRESS
	 * @param[in] key The key to read the input of. Should be GLFW_KEY_X
	 * @pre key should be a valid registered key.
	 * @return whether the key's Action is Action::PRESS
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	bool keyJustPressed(int key) const;

	/**
	 * Checks if the Action state of a keyboard key is Action::HOLD
	 * @param[in] key The key to read the input of. Should be GLFW_KEY_X
	 * @pre key should be a valid registered key.
	 * @return whether the key's Action is Action::HOLD
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	bool keyHeld(int key) const;

	/**
	 * Checks if the Action state of a keyboard key is Action::RELEASE
	 * @param[in] key The key to read the input of. Should be GLFW_KEY_X
	 * @pre key should be a valid registered key.
	 * @return whether the key's Action is Action::RELEASE
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	bool keyJustReleased(int key) const;

	/**
	 * Checks if the Action state of a keyboard key is Action::PRESS or Action::HOLD
	 * @param[in] key The key to read the input of. Should be GLFW_KEY_X
	 * @pre key should be a valid registered key.
	 * @return whether the key's Action is Action::PRESS or Action::HOLD
	 * @see https://www.glfw.org/docs/latest/group__keys.html
	 */
	bool keyPressed(int key) const;

	/**
	 * Gets the Action state of the desired mouse button key.
	 * @param[in] button The button to read the input of. Should be GLFW_MOUSE_BUTTON_X
	 * @pre button should be a valid registered mouse button.
	 * @return the Action state of the desired mouse button.
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	Action getMouse(int button) const;

	/**
	 * Checks if the Action state of a mouse button is Action::PRESS
	 * @param[in] button The mouse button to read the input of. Should be GLFW_MOUSE_BUTTON_X
	 * @pre button should be a valid registered mouse button.
	 * @return whether the mouse button's Action is Action::PRESS
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	bool mouseJustPressed(int button) const;

	/**
	 * Checks if the Action state of a mouse button is Action::HOLD
	 * @param[in] button The mouse button to read the input of. Should be GLFW_MOUSE_BUTTON_X
	 * @pre button should be a valid registered mouse button.
	 * @return whether the mouse button's Action is Action::HOLD
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	bool mouseHeld(int button) const;

	/**
	 * Checks if the Action state of a mouse button is Action::RELEASE
	 * @param[in] button The mouse button to read the input of. Should be GLFW_MOUSE_BUTTON_X
	 * @pre button should be a valid registered mouse button.
	 * @return whether the mouse button's Action is Action::RELEASE
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	bool mouseJustReleased(int button) const;

	/**
	 * Checks if the Action state of a mouse button is Action::PRESS or Action::HOLD
	 * @param[in] button The mouse button to read the input of. Should be GLFW_MOUSE_BUTTON_X
	 * @pre button should be a valid registered mouse button.
	 * @return whether the mouse button's Action is Action::PRESS or Action::HOLD
	 * @see https://www.glfw.org/docs/latest/group__buttons.html
	 */
	bool mousePressed(int button) const;

	/**
	 * Gets the position of the cursor relative to the origin of the window.
	 * @return The position of the cursor.
	 */
	glm::ivec2 getCursorPos() const;

	/**
	 * Gets how much the position of the cursor has changed since the last polling cycle.
	 * @return How much the cursor position changed.
	 */
	glm::ivec2 getCursorDelta() const;

	/**
	 * Sets the cursor position relative to the origin of the window.
	 * @param[in] cursorPos the coord to place the cursor at.
	 */
	void setCursorPos(const glm::ivec2& cursorPos);
};
