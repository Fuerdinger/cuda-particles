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

#include "Window.h"

const int WindowManager::defaultKeys[] = 
{
	GLFW_KEY_W,
	GLFW_KEY_A,
	GLFW_KEY_S,
	GLFW_KEY_D,
	GLFW_KEY_UP,
	GLFW_KEY_LEFT,
	GLFW_KEY_DOWN,
	GLFW_KEY_RIGHT,
	GLFW_KEY_SPACE,
	GLFW_KEY_ENTER,
	GLFW_KEY_ESCAPE
};

const int WindowManager::defualtMouseButtons[] =
{
	GLFW_MOUSE_BUTTON_1,
	GLFW_MOUSE_BUTTON_2,
	GLFW_MOUSE_BUTTON_3
};

void glfwErrorCallback(int error, const char* description)
{
	std::string myError;
	switch (error)
	{
	case GLFW_NO_ERROR:
		myError = "GLFW_NO_ERROR";
		break;
	case GLFW_NOT_INITIALIZED:
		myError = "GLFW_NOT_INITIALIZED";
		break;
	case GLFW_NO_CURRENT_CONTEXT:
		myError = "GLFW_NO_CURRENT_CONTEXT";
		break;
	case GLFW_INVALID_ENUM:
		myError = "GLFW_INVALID_ENUM";
		break;
	case GLFW_INVALID_VALUE:
		myError = "GLFW_INVALID_VALUE";
		break;
	case GLFW_OUT_OF_MEMORY:
		myError = "GLFW_OUT_OF_MEMORY";
		break;
	case GLFW_API_UNAVAILABLE:
		myError = "GLFW_API_UNAVAILABLE";
		break;
	case GLFW_VERSION_UNAVAILABLE:
		myError = "GLFW_VERSION_UNAVAILABLE";
		break;
	case GLFW_PLATFORM_ERROR:
		myError = "GLFW_PLATFORM_ERROR";
		break;
	case GLFW_FORMAT_UNAVAILABLE:
		myError = "GLFW_FORMAT_UNAVAILABLE";
		break;
	case GLFW_NO_WINDOW_CONTEXT:
		myError = "GLFW_NO_WINDOW_CONTEXT";
		break;
	default:
		myError = "Unknown";
		break;
	}

	fprintf(stderr, "GLFW: %s\nError: %s\n\n", description, myError.c_str());
	abort();
}

void APIENTRY glErrorCallback(GLenum source,
	GLenum type,
	unsigned int id,
	GLenum severity,
	GLsizei length,
	const char* message,
	const void* userParam)
{
	std::string mySource = "";
	std::string myType = "";
	std::string mySeverity = "";

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:
		mySource = "API";
		break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
		mySource = "Window System";
		break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER:
		mySource = "Shader Compiler";
		break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:
		mySource = "Third Party";
		break;
	case GL_DEBUG_SOURCE_APPLICATION:
		mySource = "Application";
		break;
	case GL_DEBUG_SOURCE_OTHER:
		mySource = "Other";
		break;
	default:
		mySource = "Unknown";
		break;
	}

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:
		myType = "Error";
		break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		myType = "Deprecated Behavior";
		break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		myType = "Undefined Behavior";
		break;
	case GL_DEBUG_TYPE_PORTABILITY:
		myType = "Portability";
		break;
	case GL_DEBUG_TYPE_PERFORMANCE:
		myType = "Performance";
		break;
	case GL_DEBUG_TYPE_MARKER:
		myType = "Marker";
		break;
	case GL_DEBUG_TYPE_PUSH_GROUP:
		myType = "Push Group";
		break;
	case GL_DEBUG_TYPE_POP_GROUP:
		myType = "Pop Group";
		break;
	case GL_DEBUG_TYPE_OTHER:
		myType = "Other";
		break;
	default:
		myType = "Unknown";
		break;
	}

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:
		mySeverity = "High";
		break;
	case GL_DEBUG_SEVERITY_MEDIUM:
		mySeverity = "Medium";
		break;
	case GL_DEBUG_SEVERITY_LOW:
		mySeverity = "Low";
		break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		mySeverity = "Notification";
		break;
	default:
		mySeverity = "Unknown";
		break;
	}

	fprintf(stderr, "OpenGL (%d): %s\nSource: %s\nType: %s\nSeverity: %s\n\n", id, message, mySource.c_str(), myType.c_str(), mySeverity.c_str());

	if (!(id == 131169 || id == 131185 || id == 131218 || id == 131204))
	{
		abort();
	}
}

void WindowManager::glfwCursorEnterCallback(GLFWwindow* window, int entered)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->cursorEnterCallback(entered);
}
void WindowManager::glfwWindowCloseCallback(GLFWwindow* window)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowCloseCallback();
}
void WindowManager::glfwWindowFocusCallback(GLFWwindow* window, int focused)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowFocusCallback(focused);
}
void WindowManager::glfwWindowIconifyCallback(GLFWwindow* window, int iconified)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowIconifyCallback(iconified);
}
void WindowManager::glfwFramebufferResizeCallback(GLFWwindow* window, int width, int height)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->framebufferResizeCallback(width, height);
}
void WindowManager::glfwWindowSizeCallback(GLFWwindow* window, int width, int height)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowSizeCallback(width, height);
}
void WindowManager::glfwWindowPosCallback(GLFWwindow* window, int xpos, int ypos)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowPosCallback(xpos, ypos);
}
void WindowManager::glfwWindowContentScaleCallback(GLFWwindow* window, float xscale, float yscale)
{
	WindowManager* myWindow = (WindowManager*)glfwGetWindowUserPointer(window);
	myWindow->windowContentScaleCallback(xscale, yscale);
}

void WindowManager::cursorEnterCallback(int entered)
{
	m_containsMouse = entered;
	m_cursorDelta = glm::ivec2(0.0f, 0.0f);
}
void WindowManager::windowCloseCallback()
{
	//nothing for now
}
void WindowManager::windowFocusCallback(int focused)
{
	m_focused = focused;
}
void WindowManager::windowIconifyCallback(int iconified)
{
	m_minimized = iconified;
}
void WindowManager::framebufferResizeCallback(int width, int height)
{
	m_framebufferSize = glm::ivec2(width, height);
	glViewport(0, 0, width, height);
}
void WindowManager::windowSizeCallback(int width, int height)
{
	m_size = glm::ivec2(width, height);
}
void WindowManager::windowPosCallback(int xpos, int ypos)
{
	m_pos = glm::ivec2(xpos, ypos);
}
void WindowManager::windowContentScaleCallback(float xscale, float yscale)
{
	m_scale = glm::vec2(xscale, yscale);
}


WindowManager::Action WindowManager::assignAction(Action action1, Action action2)
{
	//note that action2 can be PRESS, NONE, or CONSUMED
	//when action2 is PRESS, then the key/button has either just been pressed, or it is being held down
	//when action2 is NONE, then the key/button is not being pressed or held down
	//when action2 is CONSUMED, this is a special case which will prevent the reuse of a PRESSED key/button action

	Action ret;

	//the button/key is currently held down, and previously was either just pressed or held down
	if ((action1 == Action::PRESS || action1 == Action::HOLD) && action2 == Action::PRESS)
	{
		//either transition from a press to a hold, or continue holding
		ret = Action::HOLD;
	}
	//the button/key is not held down, and previously was just pressed or held down
	else if ((action1 == Action::PRESS || action1 == Action::HOLD) && action2 == Action::NONE)
	{
		//special state that only exists for one single polling period; button was let go
		ret = Action::RELEASE;
	}
	//the state was just consumed
	else if (action2 == Action::CONSUMED)
	{
		//only the PRESS input can be "consumed"
		//if the state is anything else, then make no change to the state
		if (action1 == Action::PRESS) ret = Action::CONSUMED;
		else ret = action1;
	}
	//the state was consumed and is now being held
	else if (action1 == Action::CONSUMED && action2 == Action::PRESS)
	{
		//skip the press state, go right to the hold state
		ret = Action::HOLD;
	}
	else
	{
		//there are no other special filtering cases
		ret = action2;
	}
	
	return ret;
}

void WindowManager::switchState(State state)
{
	switch (state)
	{
	case State::WINDOWED:
	{
		const GLFWvidmode* monitor = glfwGetVideoMode(glfwGetPrimaryMonitor());
		int width = monitor->width / 2;
		int height = monitor->height / 2;
		int xCenter = width - (width / 2);
		int yCenter = height - (height / 2);
		glfwSetWindowMonitor(m_window, nullptr, xCenter, yCenter, width, height, GLFW_DONT_CARE);
	}	break;

	case State::FULLSCREEN:
	{
		const GLFWvidmode* monitor = glfwGetVideoMode(glfwGetPrimaryMonitor());
		int width = monitor->width;
		int height = monitor->height;
		glfwSetWindowMonitor(m_window, glfwGetPrimaryMonitor(), 0, 0, width, height, GLFW_DONT_CARE);
	}	break;

	case State::BORDERLESS:
	{
		const GLFWvidmode* monitor = glfwGetVideoMode(glfwGetPrimaryMonitor());
		int width = monitor->width;
		int height = monitor->height;
		glfwSetWindowMonitor(m_window, nullptr, 0, 0, width, height, GLFW_DONT_CARE);
	}	break;

	default:
		abort();
	}

	m_state = state;
}

void WindowManager::switchMode(Mode mode)
{
	switch (mode)
	{
	case Mode::UI:
		//disable raw mouse motion, if it was enabled to begin with
		if (m_rawMotionSupported)
		{
			glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
		}

		//set the cursor to appear normally on the screen
		glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		break;

	case Mode::FPP:
		//make cursor not appear on window
		glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

		//enable raw mouse motion if supported
		if (m_rawMotionSupported)
		{
			glfwSetInputMode(m_window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
		}
		break;

	default:
		abort();
	}

	m_mode = mode;
}

WindowManager::WindowManager(const std::string& name, State state, Mode mode)
{
	//init GLFW window
	if (!glfwInit())
	{
		abort();
	}

	//set minimum required OpenGL mode to 4.3 with core profile only
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	//multisampling
	glfwWindowHint(GLFW_SAMPLES, 4);

	//debug builds should open with a debug context
#if _DEBUG
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
#endif

	glfwSetErrorCallback(glfwErrorCallback);

	//create window differently based on what kind of window we want
	int width;
	int height;
	const GLFWvidmode* vidMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
	GLFWmonitor* monitor;
	switch (state)
	{
	case State::WINDOWED:
		//draggable windows should be a quarter of the monitor size
		width = vidMode->width / 2;
		height = vidMode->height / 2;
		monitor = nullptr;
		break;
	case State::FULLSCREEN:
		width = vidMode->width;
		height = vidMode->height;
		monitor = glfwGetPrimaryMonitor();
		break;
	case State::BORDERLESS:
		width = vidMode->width;
		height = vidMode->height;
		monitor = nullptr;
		break;
	default:
		abort();
	}

	m_window = glfwCreateWindow(width, height, name.c_str(), monitor, nullptr);
	if (!m_window)
	{
		glfwTerminate();
		abort();
	}

	//set up callbacks
	glfwSetWindowUserPointer(m_window, this);
	glfwSetCursorEnterCallback(m_window, glfwCursorEnterCallback);
	glfwSetWindowCloseCallback(m_window, glfwWindowCloseCallback);
	glfwSetWindowFocusCallback(m_window, glfwWindowFocusCallback);
	glfwSetWindowIconifyCallback(m_window, glfwWindowIconifyCallback);
	glfwSetFramebufferSizeCallback(m_window, glfwFramebufferResizeCallback);
	glfwSetWindowSizeCallback(m_window, glfwWindowSizeCallback);
	glfwSetWindowPosCallback(m_window, glfwWindowPosCallback);
	glfwSetWindowContentScaleCallback(m_window, glfwWindowContentScaleCallback);

	//get window size; theoretically should always be equal to the width/height we specified earlier
	glfwGetWindowSize(m_window, &m_size.x, &m_size.y);

	//get window position; set it to the center of the screen if in windowed mode
	if (state == State::WINDOWED)
	{
		m_pos = m_size - (m_size / 2);
		glfwSetWindowPos(m_window, m_pos.x, m_pos.y);
	}
	else
	{
		glfwGetWindowPos(m_window, &m_pos.x, &m_pos.y);
	}

	//get framebuffer size, in case GLFW made it a different size than we requested, & scale
	glfwGetFramebufferSize(m_window, &m_framebufferSize.x, &m_framebufferSize.y);
	glfwGetWindowContentScale(m_window, &m_scale.x, &m_scale.y);

	//set up the window state
	m_state = state;
	switchMode(mode);
	m_focused = true;
	m_minimized = false;
	m_containsMouse = false;
	m_rawMotionSupported = glfwRawMouseMotionSupported();
	
	//register the default keys and mouse buttons for input
	int numKeys = sizeof(defaultKeys) / sizeof(int);
	for (int i = 0; i < numKeys; i++)
	{
		m_keyStates[defaultKeys[i]] = Action::NONE;
	}
	numKeys = sizeof(defualtMouseButtons) / sizeof(int);
	for (int i = 0; i < numKeys; i++)
	{
		m_mouseStates[defualtMouseButtons[i]] = Action::NONE;
	}

	//set up mouse
	m_cursorDelta = glm::ivec2(0.0f, 0.0f);
	double x, y;
	glfwGetCursorPos(m_window, &x, &y);
	m_cursorPos = glm::ivec2(floor(x), floor(y));

	//set window to be OpenGL's current context, and load OpenGL with GLAD
	glfwMakeContextCurrent(m_window);
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

	//for debug builds set up the error callback
#ifdef _DEBUG
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glErrorCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
#endif

	//enable multisampling in the GLFW framebuffer
	glEnable(GL_MULTISAMPLE);
}

WindowManager::~WindowManager()
{
	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void WindowManager::processPolls()
{
	//poll events from window system
	glfwPollEvents();

	//for each key we care about, update its Action in the map
	for (auto iter = m_keyStates.begin(); iter != m_keyStates.end(); iter++)
	{
		int state = glfwGetKey(m_window, iter->first);
		iter->second = assignAction(iter->second, (WindowManager::Action)state);
	}
	for (auto iter = m_mouseStates.begin(); iter != m_mouseStates.end(); iter++)
	{
		int state = glfwGetMouseButton(m_window, iter->first);
		iter->second = assignAction(iter->second, (WindowManager::Action)state);
	}

	//get mouse pos
	if (m_containsMouse && m_focused)
	{
		double xpos, ypos;
		glfwGetCursorPos(m_window, &xpos, &ypos);
		glm::ivec2 newPos = glm::ivec2(floor(xpos), floor(ypos));
		m_cursorDelta = newPos - m_cursorPos;
		m_cursorPos = newPos;
	}
}
void WindowManager::consumePolls()
{
	//set the special "CONSUMED" state for each value which is set to "PRESS"
	for (auto iter = m_keyStates.begin(); iter != m_keyStates.end(); iter++)
	{
		iter->second = assignAction(iter->second, Action::CONSUMED);
	}
	for (auto iter = m_mouseStates.begin(); iter != m_mouseStates.end(); iter++)
	{
		iter->second = assignAction(iter->second, Action::CONSUMED);
	}
}

void WindowManager::swapBuffers()
{
	glfwSwapBuffers(m_window);
}

void WindowManager::signalClose()
{
	glfwSetWindowShouldClose(m_window, GLFW_TRUE);
}
bool WindowManager::hasClosed() const
{
	return glfwWindowShouldClose(m_window);
}

void WindowManager::registerKey(int key)
{
	m_keyStates[key] = Action::NONE;
}
void WindowManager::registerMouseButton(int button)
{
	m_mouseStates[button] = Action::NONE;
}

WindowManager::State WindowManager::getState() const
{
	return m_state;
}
WindowManager::Mode WindowManager::getMode() const
{
	return m_mode;
}
glm::ivec2 WindowManager::getFramebufferSize() const
{
	return m_framebufferSize;
}
glm::ivec2 WindowManager::getSize() const
{
	return m_size;
}
glm::ivec2 WindowManager::getPos() const
{
	return m_pos;
}
glm::vec2 WindowManager::getScale() const
{
	return m_scale;
}

void WindowManager::setMode(WindowManager::Mode mode)
{
	if (m_mode == mode) return;
	switchMode(mode);
}
void WindowManager::setState(WindowManager::State state)
{
	if (m_state == state) return;
	switchState(state);
}
void WindowManager::setSize(const glm::ivec2& windowSize)
{
	glfwSetWindowSize(m_window, windowSize.x, windowSize.y);
	//callback will assign value
}
void WindowManager::setPos(const glm::ivec2& windowPos)
{
	glfwSetWindowPos(m_window, windowPos.x, windowPos.y);
	//callback will assign value
}

WindowManager::Action WindowManager::getKey(int key) const
{
	return m_keyStates.find(key)->second;
}
bool WindowManager::keyJustPressed(int key) const
{
	return m_keyStates.find(key)->second == Action::PRESS;
}
bool WindowManager::keyHeld(int key) const
{
	return m_keyStates.find(key)->second == Action::HOLD;
}
bool WindowManager::keyJustReleased(int key) const
{
	return m_keyStates.find(key)->second == Action::RELEASE;
}
bool WindowManager::keyPressed(int key) const
{
	Action action = m_keyStates.find(key)->second;
	return (action == Action::PRESS || action == Action::HOLD);
}

WindowManager::Action WindowManager::getMouse(int button) const
{
	return m_mouseStates.find(button)->second;
}
bool WindowManager::mouseJustPressed(int button) const
{
	return m_mouseStates.find(button)->second == Action::PRESS;
}
bool WindowManager::mouseHeld(int button) const
{
	return m_mouseStates.find(button)->second == Action::HOLD;
}
bool WindowManager::mouseJustReleased(int button) const
{
	return m_mouseStates.find(button)->second == Action::RELEASE;
}
bool WindowManager::mousePressed(int button) const
{
	Action action = m_mouseStates.find(button)->second;
	return (action == Action::PRESS || action == Action::HOLD);
}

glm::ivec2 WindowManager::getCursorPos() const
{
	return m_cursorPos;
}
glm::ivec2 WindowManager::getCursorDelta() const
{
	return m_cursorDelta;
}
void WindowManager::setCursorPos(const glm::ivec2& cursorPos)
{
	glfwSetCursorPos(m_window, cursorPos.x, cursorPos.y);
	//callback will assign value
}
