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

private:
	//see the src file for what these values do
	static const unsigned int maxNumParticles;
	static const unsigned int numParticlesToSpawn;
	static const unsigned int numParticlesAtStart;
	static const float displayParticleRadius;
	static const unsigned int displayParticleVertices;
	static const glm::vec2 particleRedRange;
	static const glm::vec2 particleGreenRange;
	static const glm::vec2 particleBlueRange;
	static const float particleRadius;
	static const pVec velocityDullingFactor;
	static const float velocityDullingFactorRate;
	static const pVec gravity;
	static const pVec boundMin;
	static const pVec boundMax;
	static const float maxExplodeRange;
	static const float maxExplodeForce;
	static const float maxSuctionRange;
	static const float maxSuctionForce;

	//number of blocks & threads we can have, at most
	unsigned int m_maxNumBlocks;
	unsigned int m_maxNumThreads;

	//ptrs to device memory for particles
	Particle* m_deviceParticlesIn;
	Particle* m_deviceParticlesOut;
	unsigned int m_numParticles;

	//graphics resources
	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ubo;
	cudaGraphicsResource_t m_uboResource;
	GLuint m_program;

	//helpers
	void swapDeviceParticles();
	pVec getCursorPos() const;
	void spawnParticles(const pVec& pos, cudaStream_t stream);

public:
	SimulationScene();
	~SimulationScene();

	void update(float deltaTime);
	void render();
	void switchFrom(const std::string& previousScene, void* data = nullptr);
};
