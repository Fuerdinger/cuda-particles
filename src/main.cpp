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

#include "SimulationScene.h"

#if defined(_WIN32) && !defined(_DEBUG)
int WINAPI wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
#else
int main(int argc, char** argv)
#endif
{
#if defined(_WIN32) && defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	//get name of config file from command line args
	std::string configFileName = "config.json";

#if defined(_WIN32) && !defined(_DEBUG)
	//note that lpCmdLine is not passed here because it does not include the path to the exe (whereas
	//GetCommandLineW() does contain it; it is required for CommandLineToArgvW to work correctly)
	int argc;
	wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	if (argc >= 2)
	{
		std::wstring wconfigFileName = std::wstring(argv[1]);
		configFileName = std::string(wconfigFileName.begin(), wconfigFileName.end());
	}
	LocalFree(argv);

#else
	if (argc >= 2)
	{
		configFileName = std::string(argv[1]);
	}
#endif
	
	//read in config file into json
	nlohmann::json config;
	try
	{
		std::fstream configFile(configFileName, std::ios::in);
		if (!configFile.good()) throw std::exception(("Could not open config file named " + configFileName).c_str());
		config = nlohmann::json::parse(configFile, nullptr, true, true);
		configFile.close();
	}
	catch (std::exception& e)
	{
#if defined(_WIN32) && !defined(_DEBUG)
		MessageBox(NULL, e.what(), "Error", MB_OK);
#else
		fprintf(stderr, "%s\n", e.what());
#endif
		abort();
	}

	//run program using config data
	SceneManager manager;
	manager.build();
	manager.run({ new SimulationScene(config) });
}
