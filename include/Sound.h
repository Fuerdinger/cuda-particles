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
 * @brief A struct for specifying SoundPlayer values at construction.
 * @details The default values of the struct mimic the default values of
 * SFML audio sources. This struct does not need to be used explicitly
 * to construct SoundPlayers; instead, users may change any of the settings
 * after construction.
 *
 * @see SoundManager::createSoundPlayer(const std::string&, SoundSettings*);
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
struct SoundSettings
{
	//if position or attenuation are other than the default values,
	//then the sound will be interpreted to not be relative to the player.

	/// The 3D coordinate where the audio lives in 3D space
	glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
	/// How loud the audio should be. 1.0 = quiet, 100.0 = loud
	float volume = 100.0f;
	/// 1.0 = fades slowly as you move away, 100.0 = fades quickly as you move away
	float attenuation = 1.0f;
	/// If the player is closer than this distance, they will hear the audio at max volume.
	float minDistance = 1.0f;
	/// The pitch of the audio. 1.0 = normal, > 1.0 = fast, < 1.0 = slow
	float pitch = 1.0f;
	/// Whether the audio should play again once it finishes the first time.
	bool loop = false;
	/// Whether or not the 3D position of a SoundPlayer is relative to the player (true) or absolute (false).
	bool playerRelative = false;
};


class SoundPlayer;

/**
 * @brief A class that maintains an audio library and produces SoundPlayer instances to play the audio.
 * @details A static pointer to an instance of this class is accessible
 * to Scene instances, allowing the sharing of audio across them.
 * This class may be explicitly initialized by a user of SceneManager
 * to allow for sound loading at startup. Currently, only OGG sound files may be loaded.
 *
 * @pre Only one SoundManager should be instantiated in the application.
 *
 * @see SoundPlayer
 * @see Scene::_sounds
 * @see SceneManager::createSounds(SoundManager*)
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, under the MIT License
 */
class SoundManager
{
private:
	std::map<std::string, sf::SoundBuffer*> m_soundBuffers;
	std::map<std::string, sf::Music*> m_songs;
	std::set<SoundPlayer*> m_soundPlayers;
	static std::string _path;

public:
	/**
	 * Constructs a SoundManager with no audio loaded in.
	 */
	SoundManager();

	/**
	 * Constructs a SoundManager with audio loaded in. The .ogg file extension is assumed.
	 * @param[in] sfx The names of the audio files to load in as SFX (no extension)
	 * @param[in] music The names of the audio files to load in as music (no extension)
	 * @pre sfx and music do not contain duplicate strings.
	 * @see setPath(const std::string&) for setting the directory where the audio resides.
	 */
	SoundManager(const std::vector<std::string>& sfx, const std::vector<std::string>& music);

	/**
	 * Frees all sounds, as well as all SoundPlayers not cleaned up by users.
	 */
	~SoundManager();

	/**
	 * Loads a OGG file as SFX. The .ogg file extension is assumed.
	 * SFX will be loaded into CPU memory in its entirety when this function is called.
	 * @param[in] name The name of the audio file to load in (no extension)
	 * @pre the audio file has not already been loaded in as either sfx or music.
	 * @pre the audio file actually exists, and must reside at path + name + .ogg
	 * @see setPath(const std::string&) for setting the directory where the audio resides.
	 */
	void loadSFX(const std::string& name);

	/**
	 * Loads a OGG file as music. The .ogg file extension is assumed.
	 * Music will be streamed into CPU memory when a SoundPlayer attempts to play it.
	 * @param[in] name The name of the audio file to load in (no extension)
	 * @pre the audio file has not already been loaded in as either sfx or music.
	 * @pre the audio file actually exists, and must reside at path + name + .ogg
	 * @see setPath(const std::string&) for setting the directory where the audio resides.
	*/
	void loadMusic(const std::string& name);

	/**
	 * Creates a SoundPlayer to play the selected SFX or music.
	 * The user should not call delete or free on it. They should instead either let the SoundManager
	 * free the resource on its own, or should explicitly call deleteSoundPlayer(SoundPlayer*) to free it.
	 * @param[in] name The name of the audio file to create a SoundPlayer for
	 * @param[in] settings The settings to apply to the SoundPlayer; if not provided, the default values will be set
	 * @return pointer to a newly created SoundPlayer for the specified audio, or nullptr if such audio doesn't exist
	 */
	SoundPlayer* createSoundPlayer(const std::string& name, SoundSettings* settings = nullptr);

	/**
	 * Deletes a SoundPlayer.
	 * The user does not need to call this; when the SoundManager which created the SoundPlayer is
	 * destructed, it will clean up all SoundPlayer instances it created which the user did not clean up.
	 * @param[in] soundPlayer Pointer to the SoundPlayer to clean up.
	 * @pre soundPlayer must be a pointer to a valid SoundPlayer, which has not been cleaned up already.
	 */
	void deleteSoundPlayer(SoundPlayer* soundPlayer);

	/**
	 * Sets the directory for where audio files should be loaded from.
	 * The default path is empty, meaning that audio files are expected to be in the same
	 * directory as the executable until this function is called with a different path.
	 * @param[in] path The directory of the audio files. If not empty, should end with a slash.
	 * @see SoundManager(const std::vector<std::string>&, const std::vector<std::string>&)
	 * @see loadSFX(const std::string&)
	 * @see loadMusic(const std::string&)
	 */
	static void setPath(const std::string& path);
};

/**
 * @brief A class that wraps around an SFML audio source.
 * @details Obtaining an instance of this class can only be done via
 * a SoundManager, which will produce a SoundPlayer which allows playback
 * of an audio source loaded into the SoundManager. Multiple SoundPlayer instances
 * which wrap around the same audio loaded as an SFX may coexist harmoniously, but 
 * multiple SoundPlayer instances which wrap around the same audio loaded as music
 * may cause issues if they both attempt to play the music at the same time.
 *
 * @see SoundManager::createSoundPlayer(const std::string&, SoundSettings* settings);
 *
 * @author Daniel Fuerlinger
 * @date 2022
 * @copyright 2022 Daniel Fuerlinger, all rights reserved
 */
class SoundPlayer
{
	friend SoundManager;
private:
	std::string m_name;
	sf::SoundSource* m_sound;
	bool m_isMusic;

	//constructor loads the sound in from memory from assets\\sound\\name
	//only SoundManager can construct a SoundPlayer
	SoundPlayer(const std::string& sfxName, sf::SoundBuffer* buf, SoundSettings* settings);
	SoundPlayer(const std::string& musicName, sf::Music* music, SoundSettings* settings);
public:
	/**
	 * Destructor. Frees all memory allocated by the SoundPlayer. Do not delete SoundPlayers
	 * yourself; instead, let the SoundManager delete them on its own, or explicitly call
	 * SoundManager::deleteSoundPlayer(SoundPlayer*)
	 */
	~SoundPlayer();

	/**
	 * Getter for the name of the audio which the SoundPlayer plays.
	 * @return The name of the audio
	 */
	std::string getName() const;

	/**
	 * Sets all properties of the SoundPlayer using SoundSettings struct.
	 * @param[in] settings The settings to apply to the SoundPlayer
	 */
	void set(SoundSettings* settings);

	/**
	 * Sets the loop property of the SoundPlayer.
	 * @param[in] loop Set to true if you want the audio to loop
	 */
	void setLoop(bool loop);

	/**
	 * Sets the volume property of the SoundPlayer.
	 * @param[in] volume How loud the audio should be. 1.0 = quiet, 100.0 = loud
	 */
	void setVolume(float volume);

	/**
	 * Sets the pitch property of the SoundPlayer.
	 * @param[in] pitch > 1.0 is high pitch, < 1.0 is low pitch
	 */
	void setPitch(float pitch);

	/**
	 * Sets the attenuation property of the SoundPlayer. Has no effect
	 * unless the audio is mono.
	 * @param[in] attenuation 1.0 = fades slowly as you move away, 100.0 = fades quickly as you move away
	 */
	void setAttenuation(float attenuation);

	/**
	 * Sets the mininimum distance property of the SoundPlayer. Has no
	 * effect unless the audio is mono.
	 * @param[in] minDistance If the player is closer than this distance, they will hear the audio at max volume.
	 */
	void setMinDistance(float minDistance);

	/**
	 * Sets the position property of the SoundPlayer. Has no effect unless
	 * the audio is mono.
	 * @param[in] position The 3D position where the audio should reside.
	 */
	void setPosition(const glm::vec3& position);

	/**
	 * Sets the player relative property of the SoundPlayer. Has no effect
	 * unless the audio is mono.
	 * @param[in] relative True if the 3D position of the audio should be relative to
	 * the player position, False if the position should be absolute
	 */
	void setPlayerRelative(bool relative);

	//getters for all sound properties

	/**
	 * Gets the loop property of the SoundPlayer.
	 * @return True if the audio loops, False if it doesn't
	 */
	bool getLoop() const;

	/**
	 * Gets the volume property of the SoundPlayer.
	 * @return How loud the sound is
	 */
	float getVolume() const;

	/**
	 * Gets the pitch property of the SoundPlayer
	 * @return How high/low pitch the sound is
	 */
	float getPitch() const;

	/**
	 * Gets the attenuation property of the SoundPlayer
	 * @return How quickly/slowly the audio fades as the player draws near it
	 */
	float getAttenuation() const;

	/**
	 * Gets the minimum distance property of the SoundPlayer
	 * @return How close the the player needs to be to hear the audio at its max volume
	 */
	float getMinDistance() const;

	/**
	 * Gets the position property of the SoundPlayer.
	 * @return the 3D coordinate where the audio lives in 3D space
	 */
	glm::vec3 getPosition() const;

	/**
	 * Gets the player relative property of the SoundPlayer
	 * @return True if the audio's 3D position is relative to the player,
	 * false if it is absolute.
	 */
	bool getPlayerRelative() const;

	/**
	 * Gets whether the audio is currently playing or not
	 * @return True if the audio is currently playing, false if it is paused or stopped
	 */
	bool isPlaying() const;

	/**
	 * Begins playback of the audio. If the audio had been paused previously,
	 * the audio will resume where it was paused. If the audio had been stopped
	 * previously, then the audio will resume at the beginning.
	 */
	void play();

	/**
	 * Pauses the playback of the audio. If the audio was not playing to begin with,
	 * then this has no effect.
	 */
	void pause();

	/**
	 * Stops the playback of the audio. If the audio was not playing to begin with,
	 * then this has no effect.
	 */
	void stop();
};
