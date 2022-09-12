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
#include "Scene.h"

class SimulationScene : public Scene
{
public:
	//number of dimensions of the particles in the sim
	typedef glm::vec2 pVec;

	//data for each particle in the sim
	struct Particle
	{
		pVec position;
		pVec velocity;
		glm::vec4 color;
	};

	struct Collision
	{
		float force;
	};

private:
	//see config.json for what these values do
	struct Config
	{
		unsigned int maxNumParticles;
		unsigned int numParticlesToSpawn;
		unsigned int numParticlesAtStart;
		float displayParticleRadius;
		unsigned int displayParticleVertices;
		glm::vec2 particleRedRange;
		glm::vec2 particleGreenRange;
		glm::vec2 particleBlueRange;
		float particleRadius;
		pVec velocityDullingFactor;
		float velocityDullingFactorRate;
		float restitutionCoefficient;
		pVec gravity;
		pVec boundMin;
		pVec boundMax;
		float maxExplodeRange;
		float maxExplodeForce;
		float maxSuctionRange;
		float maxSuctionForce;
		bool audioOn;
		std::string audioFilePrefix;
		unsigned int numSoundPlayers;
		float maxPitch;
		float minPitch;
		float minPitchForce;
		Config(nlohmann::json& json);
	} const m_cfg;

	//number of blocks & threads we can have, at most
	unsigned int m_maxNumBlocks;
	unsigned int m_maxNumThreads;

	//ptrs to device memory for particles
	Particle* m_deviceParticlesIn;
	Particle* m_deviceParticlesOut;
	unsigned int m_numParticles;

	//CPU buffer of particles for spawning
	Particle* m_newParticles;

	//CPU and GPU buffer for collision data
	Collision* m_deviceCollisions;
	Collision* m_collisions;

	//graphics resources
	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ssbo;
	cudaGraphicsResource_t m_ssboResource;
	GLuint m_program;

	//SFX players
	std::vector<SoundPlayer*> m_sounds;

	//helpers
	void swapDeviceParticles();
	pVec getCursorPos() const;
	void spawnParticles(const pVec& pos, cudaStream_t stream);

public:
	SimulationScene(nlohmann::json& config);
	~SimulationScene();

	void update(float deltaTime);
	void render();
	void switchFrom(const std::string& previousScene, void* data = nullptr);
};
