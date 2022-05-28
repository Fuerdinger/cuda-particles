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

#include "Sound.h"

std::string SoundManager::_path = "";

SoundManager::SoundManager()
{

}

SoundManager::SoundManager(const std::vector<std::string>& sfx, const std::vector<std::string>& music)
{
	for (auto iter = sfx.begin(); iter != sfx.end(); iter++)
	{
		loadSFX(*iter);
	}

	for (auto iter = music.begin(); iter != music.end(); iter++)
	{
		loadMusic(*iter);
	}
}

//frees memory for all loaded assets
SoundManager::~SoundManager()
{
	for (auto iter = m_soundPlayers.begin(); iter != m_soundPlayers.end(); iter++)
	{
		delete (*iter);
	}
	m_soundPlayers.clear();

	for (auto iter = m_soundBuffers.begin(); iter != m_soundBuffers.end(); iter++)
	{
		delete iter->second;
	}
	m_soundBuffers.clear();

	for (auto iter = m_songs.begin(); iter != m_songs.end(); iter++)
	{
		delete iter->second;
	}
	m_songs.clear();
}

//loads the sound in from memory from the path + name
void SoundManager::loadSFX(const std::string& name)
{
	assert(m_soundBuffers.find(name) == m_soundBuffers.end());
	sf::SoundBuffer* buf = new sf::SoundBuffer();
	bool ret = buf->loadFromFile(_path + name + ".ogg");
	assert(ret);
	m_soundBuffers[name] = buf;
}
void SoundManager::loadMusic(const std::string& name)
{
	assert(m_songs.find(name) == m_songs.end());
	sf::Music* music = new sf::Music();
	bool ret = music->openFromFile(_path + name + ".ogg");
	assert(ret);
	m_songs[name] = music;
}


//allows client to play a particular sound file via SoundPlayer class
//SoundManager will clean up all alloc'd soundplayers.
SoundPlayer* SoundManager::createSoundPlayer(const std::string& name, SoundSettings* settings)
{
	//if the passed in settings are null, then use default values
	SoundSettings mySettings;
	if (settings == nullptr)
	{
		settings = &mySettings;
	}

	SoundPlayer* player;

	//if it's an SFX, then construct an SFX soundplayer
	auto iter = m_soundBuffers.find(name);
	if (iter != m_soundBuffers.end())
	{
		sf::SoundBuffer* buf = m_soundBuffers[name];
		player = new SoundPlayer(name, buf, settings);
	}
	//if it's a song, then construct a music soundplayer
	else
	{
		if (m_songs.find(name) == m_songs.end()) return nullptr;
		sf::Music* song = m_songs[name];
		player = new SoundPlayer(name, song, settings);
	}

	m_soundPlayers.insert(player);

	return player;
}

//the client has the option of cleaning up SoundPlayers themselves.
void SoundManager::deleteSoundPlayer(SoundPlayer* soundPlayer)
{
	auto iter = m_soundPlayers.find(soundPlayer);
	assert(iter != m_soundPlayers.end());
	delete (*iter);
	m_soundPlayers.erase(iter);
}

void SoundManager::setPath(const std::string& path)
{
	_path = path;
}


//constructor loads the sound in from memory from assets\\sounds\\name
SoundPlayer::SoundPlayer(const std::string& sfxName, sf::SoundBuffer* buf, SoundSettings* settings)
	: m_name(sfxName), m_isMusic(false)
{
	m_sound = new sf::Sound();
	((sf::Sound*)m_sound)->setBuffer(*buf);

	set(settings);
}
SoundPlayer::SoundPlayer(const std::string& musicName, sf::Music* music, SoundSettings* settings)
	: m_name(musicName), m_isMusic(true)
{
	m_sound = music;

	set(settings);
}

//frees memory allocated for sound
SoundPlayer::~SoundPlayer()
{
	stop();

	//music is deleted by the SoundManager
	if (!m_isMusic)
	{
		delete m_sound;
	}
}

std::string SoundPlayer::getName() const
{
	return m_name;
}

void SoundPlayer::set(SoundSettings* settings)
{
	setLoop(settings->loop);
	setVolume(settings->volume);
	setPitch(settings->pitch);
	setAttenuation(settings->attenuation);
	setMinDistance(settings->minDistance);
	setPosition(settings->position);
	setPlayerRelative(settings->playerRelative);
}

void SoundPlayer::setLoop(bool loop)
{
	if (m_isMusic)
	{
		((sf::Music*)m_sound)->setLoop(loop);
	}
	else
	{
		((sf::Sound*)m_sound)->setLoop(loop);
	}
}
void SoundPlayer::setVolume(float volume)
{
	m_sound->setVolume(volume);
}
void SoundPlayer::setPitch(float pitch)
{
	m_sound->setPitch(pitch);
}
void SoundPlayer::setAttenuation(float attenuation)
{
	m_sound->setAttenuation(attenuation);
}
void SoundPlayer::setMinDistance(float minDistance)
{
	m_sound->setMinDistance(minDistance);
}
void SoundPlayer::setPosition(const glm::vec3& position)
{
	m_sound->setPosition(sf::Vector3(position.x, position.y, position.z));
}
void SoundPlayer::setPlayerRelative(bool relative)
{
	m_sound->setRelativeToListener(relative);
}

bool SoundPlayer::getLoop() const
{
	if (m_isMusic)
	{
		return ((sf::Music*)m_sound)->getLoop();
	}
	else
	{
		return ((sf::Sound*)m_sound)->getLoop();
	}
}
float SoundPlayer::getVolume() const
{
	return m_sound->getVolume();
}
float SoundPlayer::getPitch() const
{
	return m_sound->getPitch();
}
float SoundPlayer::getAttenuation() const
{
	return m_sound->getAttenuation();
}
float SoundPlayer::getMinDistance() const
{
	return m_sound->getMinDistance();
}
glm::vec3 SoundPlayer::getPosition() const
{
	sf::Vector3 pos = m_sound->getPosition();
	return glm::vec3(pos.x, pos.y, pos.z);
}
bool SoundPlayer::getPlayerRelative() const
{
	return m_sound->isRelativeToListener();
}
bool SoundPlayer::isPlaying() const
{
	return m_sound->getStatus() == sf::Sound::Status::Playing;
}

//starts sound playback
void SoundPlayer::play()
{
	m_sound->play();
}

//stops sound playback; next playback will resume where it stopped
void SoundPlayer::pause()
{
	m_sound->pause();
}

//stops sound playback; next playback will begin at start
void SoundPlayer::stop()
{
	m_sound->stop();
}
