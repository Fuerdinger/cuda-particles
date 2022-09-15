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

#define PVEC_DIM (sizeof(SimulationScene::pVec) / sizeof(float))

std::mt19937_64 rng;

void cudaErrorCheck(cudaError_t result, const char* func, int line, const char* file)
{
#ifndef NDEBUG
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA ERROR!\nFunction: %s\nLine: %d\nFile: %s\nError Name: %s\nError Description: %s\n",
			func, line, file, cudaGetErrorName(result), cudaGetErrorString(result));
		abort();
	}
#endif
}
#define cuCheck(x) cudaErrorCheck(x, #x, __LINE__, __FILE__)
#define cuAsyncCheck() cuCheck(cudaDeviceSynchronize())

__constant__ float d_particleRadius;
__constant__ SimulationScene::pVec d_velocityDullingFactor;
__constant__ float d_velocityDullingFactorRate;
__constant__ float d_restitutionCoefficient;
__constant__ SimulationScene::pVec d_gravity;
__constant__ SimulationScene::pVec d_boundMin;
__constant__ SimulationScene::pVec d_boundMax;
__shared__ SimulationScene::pVec d_velocityDelta;
__shared__ SimulationScene::pVec d_positionDelta;
__shared__ float d_collisionForceDelta;

__device__ float cudaDotProduct(const SimulationScene::pVec& particle1, const SimulationScene::pVec& particle2)
{
	float sum = 0.0f;
	#pragma unroll
	for (char i = 0; i < PVEC_DIM; i++)
	{
		sum += particle1[i] * particle2[i];
	}
	return sum;
}

__device__ SimulationScene::pVec cudaPow(const SimulationScene::pVec& base, const float power)
{
	SimulationScene::pVec ret = SimulationScene::pVec(0.0f);
	#pragma unroll
	for (char i = 0; i < PVEC_DIM; i++)
	{
		ret[i] = powf(base[i], power);
	}
	return ret;
}

__device__ float cudaCeil(const float val)
{
	return (float)((unsigned int)(val + 1));
}

__device__ unsigned int cudaMin(const unsigned int val1, const unsigned int val2)
{
	return val1 < val2 ? val1 : val2;
}

__global__ void cudaRun(
	const SimulationScene::Particle* const particlesIn,
	SimulationScene::Particle* const particlesOut,
	SimulationScene::Collision* const collisions,
	const unsigned int numParticles,
	const float deltaTime)
{
	if (threadIdx.x == 0) 
	{ 
		d_velocityDelta = SimulationScene::pVec(0.0f);
		d_positionDelta = SimulationScene::pVec(0.0f);
		d_collisionForceDelta = 0.0f;
	}
	__syncthreads();

	const unsigned int index = blockIdx.x;
	const SimulationScene::Particle in = particlesIn[index];
	SimulationScene::Particle* const out = particlesOut + index;
	SimulationScene::Collision* const collision = collisions + index;

	const SimulationScene::pVec inVelocity = in.velocity;
	const SimulationScene::pVec inPosition = in.position + inVelocity * deltaTime;

	//get particles to check to check for collisions
	const unsigned int numOtherParticles = cudaCeil((float)numParticles / (float)blockDim.x);
	const unsigned int firstOtherParticle = threadIdx.x * numOtherParticles;
	const unsigned int lastOtherParticle = cudaMin(firstOtherParticle + numOtherParticles, numParticles);

	SimulationScene::pVec velocityDelta = SimulationScene::pVec(0.0f);
	SimulationScene::pVec positionDelta = SimulationScene::pVec(0.0f);
	float collisionDelta = 0.0f;

	//compute velocity/position based on colliding particles

	for (unsigned int i = firstOtherParticle; i < lastOtherParticle; i++)
	{
		if (i == index) continue;
		const SimulationScene::Particle other = particlesIn[i];

		//if the distance between the two circle centerpoints is less than the sum
		//of the radii, then they must be overlapping.
		const SimulationScene::pVec otherVelocity = other.velocity;
		const SimulationScene::pVec otherPos = other.position + otherVelocity * deltaTime;
		const SimulationScene::pVec delta = inPosition - otherPos;
		const float distanceSqr = cudaDotProduct(delta, delta);
		const float radiusSum = d_particleRadius + d_particleRadius;
		const float radiusSumSqr = radiusSum * radiusSum;

		//to avoid computing a square root, the squared values are compared instead
		if (distanceSqr < radiusSumSqr)
		{
			//if an overlap occurs, move the particle out of the way by half the overlapping distance
			const float distance = sqrt(distanceSqr);
			const float overlappingDistance = radiusSum - distance;
			const SimulationScene::pVec overlappingDelta = (delta / distance) * overlappingDistance;
			const SimulationScene::pVec dis = overlappingDelta * 0.5f;
			positionDelta += dis;

			//if the particles are sufficiently far enough away, update velocities too
			if (distanceSqr > 0.001f)
			{
				//elastic collision formula: https://en.wikipedia.org/wiki/Elastic_collision
				const float velocityMag = cudaDotProduct(inVelocity - otherVelocity, delta) / distanceSqr;
				velocityDelta += d_restitutionCoefficient * -velocityMag * delta;
				collisionDelta += velocityMag > 0.0f ? velocityMag : -1.0f * velocityMag;
			}
		}
	}
	#pragma unroll
	for (char i = 0; i < PVEC_DIM; i++)
	{
		atomicAdd(&d_velocityDelta[i], velocityDelta[i]);
		atomicAdd(&d_positionDelta[i], positionDelta[i]);
	}
	atomicAdd(&d_collisionForceDelta, collisionDelta);
	__syncthreads();

	//only 1 thread should do this part
	if (threadIdx.x == 0)
	{
		//velocity becomes the summed result of the potential elastic collisions
		SimulationScene::pVec outVelocity = inVelocity + d_velocityDelta;

		//dull the velocity and apply force of gravity
		outVelocity *= cudaPow(d_velocityDullingFactor, deltaTime * d_velocityDullingFactorRate);
		outVelocity += d_gravity * deltaTime;

		//position calculated based on velocity and sum of overlaps
		SimulationScene::pVec outPosition = in.position + outVelocity * deltaTime + d_positionDelta;

		//ensure the particle hasn't left the simulation bounds
		#pragma unroll
		for (char i = 0; i < PVEC_DIM; i++)
		{
			//not completely elastic; is dulled
			if (outPosition[i] < d_boundMin[i])
			{
				outPosition[i] = d_boundMin[i]; 
				outVelocity[i] *= -d_restitutionCoefficient;
				d_collisionForceDelta += sqrt(outVelocity[i] * outVelocity[i]);
			}
			else if (outPosition[i] > d_boundMax[i])
			{
				outPosition[i] = d_boundMax[i];
				outVelocity[i] *= -d_restitutionCoefficient;
				d_collisionForceDelta += sqrt(outVelocity[i] * outVelocity[i]);
			}
		}

		//finally store results
		out->position = outPosition;
		out->velocity = outVelocity;
		out->color = in.color;
		collision->force = d_collisionForceDelta;
	}
}

__device__ float cudaLerp(const float x0, const float x1, const float t)
{
	return x0 + t * (x1 - x0);
}

float lerp(const float x0, const float x1, const float t)
{
	return x0 + t * (x1 - x0);
}

__constant__ float d_maxExplodeRange;
__constant__ float d_maxExplodeForce;

__global__ void cudaExplode(
	SimulationScene::Particle* const inParticles,
	const unsigned int numParticles,
	const float deltaTime,
	const SimulationScene::pVec pos)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;
	SimulationScene::Particle* const in = inParticles + index;

	//apply a velocity based on the vector from the explosion to particle
	const SimulationScene::pVec dis = in->position - pos;
	const float distanceSqr = cudaDotProduct(dis, dis);
	if (distanceSqr < d_maxExplodeRange * d_maxExplodeRange)
	{
		//force is added (rather than assigned) to ensure realism of particles far away from explosion
		const float force = cudaLerp(d_maxExplodeForce, 0.0f, sqrt(distanceSqr) / d_maxExplodeRange);
		in->velocity += dis * force;
	}
}

__constant__ float d_maxSuctionRange;
__constant__ float d_maxSuctionForce;

__global__ void cudaSuction(
	SimulationScene::Particle* const inParticles,
	const unsigned int numParticles,
	const float deltaTime,
	const SimulationScene::pVec pos)
{
	const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles) return;
	SimulationScene::Particle* const in = inParticles + index;

	//apply a velocity based on the vector from the suction to particle
	const SimulationScene::pVec dis = in->position - pos;
	const float distanceSqr = cudaDotProduct(dis, dis);
	if (distanceSqr < d_maxSuctionRange * d_maxSuctionRange)
	{
		//force is assigned (rather than added) to prevent particle velocities from getting too high over time
		const float force = cudaLerp(d_maxSuctionForce, 0.0f, sqrt(distanceSqr) / d_maxSuctionRange);
		in->velocity = -dis * force;
	}
}

void SimulationScene::swapDeviceParticles()
{
	Particle* const temp = m_deviceParticlesIn;
	m_deviceParticlesIn = m_deviceParticlesOut;
	m_deviceParticlesOut = temp;
}

SimulationScene::pVec SimulationScene::getCursorPos() const
{
	glm::vec2 cursorPos = _window->getCursorPos();
	cursorPos.x -= _window->getFramebufferSize().x * 0.5f;
	cursorPos.y -= _window->getFramebufferSize().y * 0.5f;
	cursorPos.y *= -1.0f;
	pVec pos = cursorPos;
	return pos;
}

SimulationScene::Config::Config(nlohmann::json& json)
{
	maxNumParticles = json["maxNumParticles"];
	numParticlesToSpawn = json["numParticlesToSpawn"];
	numParticlesAtStart = json["numParticlesAtStart"];
	displayParticleRadius = json["displayParticleRadius"];
	displayParticleVertices = json["displayParticleVertices"];
	velocityColorLimit = json["velocityColorLimit"];

	for (char i = 0; i < 2; i++)
	{
		particleRedRange[i] = json["particleRedRange"][i];
		particleGreenRange[i] = json["particleGreenRange"][i];
		particleBlueRange[i] = json["particleBlueRange"][i];
	}

	for (char i = 0; i < PVEC_DIM; i++)
	{
		velocityDullingFactor[i] = json["velocityDullingFactor"][i];
		gravity[i] = json["gravity"][i];
		boundMin[i] = json["boundMin"][i];
		boundMax[i] = json["boundMax"][i];
	}

	for (char i = 0; i < 4; i++)
	{
		velocityColor[i] = json["velocityColor"][i];
	}

	particleRadius = json["particleRadius"];
	velocityDullingFactorRate = json["velocityDullingFactorRate"];
	restitutionCoefficient = json["restitutionCoefficient"];

	maxExplodeRange = json["maxExplodeRange"];
	maxExplodeForce = json["maxExplodeForce"];
	maxSuctionRange = json["maxSuctionRange"];
	maxSuctionForce = json["maxSuctionForce"];
	audioOn = json["audioOn"];
	audioFilePrefix = json["audioFilePrefix"];
	numSoundPlayers = json["numSoundPlayers"];
	maxPitch = json["maxPitch"];
	minPitch = json["minPitch"];
	minPitchForce = json["minPitchForce"];
}

SimulationScene::SimulationScene(nlohmann::json& config)
	: Scene("Simulation")
	, m_cfg(config)
{
	//load audio files
	_sounds->setPath("sfx\\");
	std::vector<std::string> soundNames;
	for (const auto& file : std::filesystem::directory_iterator("sfx"))
	{
		std::filesystem::path path = file.path();
		if (path.extension() == ".ogg")
		{
			std::string name = path.stem().string();
			if (name.find(m_cfg.audioFilePrefix) != std::string::npos)
			{
				_sounds->loadSFX(name);
				soundNames.push_back(name);
			}
		}
	}
	//create sound players for the audio files
	const unsigned int numSoundPlayers = m_cfg.numSoundPlayers > 0 ? m_cfg.numSoundPlayers : soundNames.size();
	m_sounds.reserve(numSoundPlayers);
	for (unsigned int i = 0; i < numSoundPlayers; i++)
	{
		std::string soundName = soundNames[i % soundNames.size()];
		m_sounds.push_back(_sounds->createSoundPlayer(soundName));
	}

	//generate distributions based on size of simulation bounding box, expected color values
	std::uniform_real_distribution<float> dis[PVEC_DIM];
	for (char i = 0; i < PVEC_DIM; i++)
	{
		dis[i] = std::uniform_real_distribution<float>(m_cfg.boundMin[i], m_cfg.boundMax[i]);
	}
	std::uniform_real_distribution<float> redDis = std::uniform_real_distribution<float>(m_cfg.particleRedRange.x, m_cfg.particleRedRange.y);
	std::uniform_real_distribution<float> greenDis = std::uniform_real_distribution<float>(m_cfg.particleGreenRange.x, m_cfg.particleGreenRange.y);
	std::uniform_real_distribution<float> blueDis = std::uniform_real_distribution<float>(m_cfg.particleBlueRange.x, m_cfg.particleBlueRange.y);

	m_numParticles = m_cfg.numParticlesAtStart;

	Particle* particles = (Particle*)malloc(m_cfg.maxNumParticles * sizeof(Particle));
	m_newParticles = (Particle*)malloc(m_cfg.numParticlesToSpawn * sizeof(Particle));
	m_collisions = (Collision*)malloc(m_cfg.maxNumParticles * sizeof(Collision));

	//initialize random positions and colors
	for (unsigned int i = 0; i < m_numParticles; i++)
	{ 
		for (char j = 0; j < PVEC_DIM; j++)
		{
			particles[i].position[j] = dis[j](rng);
		}
		particles[i].velocity = pVec(0.0f);
		particles[i].color = glm::vec4(redDis(rng), greenDis(rng), blueDis(rng), 1.0f);
		m_collisions[i].force = 0.0f;
	};

	//put the particle data into a SSBO
	glGenBuffers(1, &m_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, m_cfg.maxNumParticles * sizeof(Particle), particles, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//generate triangle fan of circle vertices
	glm::vec2* circlePoints = (glm::vec2*)malloc(m_cfg.displayParticleVertices * sizeof(glm::vec2));
	float angleInterval = 6.283185f / (m_cfg.displayParticleVertices - 2);
	circlePoints[0] = glm::vec2(0.0f, 0.0f);
	for (unsigned int i = 1; i < m_cfg.displayParticleVertices; i++)
	{
		const float angle = (i - 1) * angleInterval;
		circlePoints[i] = glm::vec2(cos(angle) * m_cfg.displayParticleRadius, sin(angle) * m_cfg.displayParticleRadius);
	}

	//VAO with 1 VBOs for the circle
	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);
	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, m_cfg.displayParticleVertices * sizeof(glm::vec2), circlePoints, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	free(circlePoints);

	//cuda can write to the particle vertices via m_ssboResource
	cuCheck(cudaGraphicsGLRegisterBuffer(&m_ssboResource, m_ssbo, cudaGraphicsRegisterFlagsWriteDiscard));

	//allocate memory for cuda particles, copy data into the first one
	cuCheck(cudaMalloc((void**)&m_deviceParticlesIn, m_cfg.maxNumParticles * sizeof(Particle)));
	cuCheck(cudaMalloc((void**)&m_deviceParticlesOut, m_cfg.maxNumParticles * sizeof(Particle)));
	cuCheck(cudaMemcpy(m_deviceParticlesIn, particles, m_cfg.maxNumParticles * sizeof(Particle), cudaMemcpyHostToDevice));
	free(particles);

	//allocate memory for collision data
	cuCheck(cudaMalloc((void**)&m_deviceCollisions, m_cfg.maxNumParticles * sizeof(Collision)));
	cuCheck(cudaMemcpy(m_deviceCollisions, m_collisions, m_cfg.maxNumParticles * sizeof(Collision), cudaMemcpyHostToDevice));

	//copy simulation constants into constant cuda memory
	cuCheck(cudaMemcpyToSymbol(d_particleRadius, &m_cfg.particleRadius, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_velocityDullingFactor, &m_cfg.velocityDullingFactor, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_velocityDullingFactorRate, &m_cfg.velocityDullingFactorRate, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_restitutionCoefficient, &m_cfg.restitutionCoefficient, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_gravity, &m_cfg.gravity, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_boundMin, &m_cfg.boundMin, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_boundMax, &m_cfg.boundMax, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxExplodeRange, &m_cfg.maxExplodeRange, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxExplodeForce, &m_cfg.maxExplodeForce, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxSuctionRange, &m_cfg.maxSuctionRange, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxSuctionForce, &m_cfg.maxSuctionForce, sizeof(float), 0, cudaMemcpyHostToDevice));

	//get properties from user's GPU to ensure maximum performance of simulation
	cudaDeviceProp properties;
	cuCheck(cudaGetDeviceProperties(&properties, 0));
	m_maxNumBlocks = properties.maxGridSize[0];
	m_maxNumThreads = properties.maxThreadsDim[0];
	if (m_maxNumBlocks < m_cfg.maxNumParticles)
	{
		fprintf(stderr, "Error: gpu's max grid size is %d, which is too low for %d particles", m_maxNumBlocks, m_cfg.maxNumParticles);
		abort();
	}

	//create rendering program from shaders
	m_program = _util->createProgramFromDisk("shaders\\Vert.shader", "shaders\\Frag.shader");

	//upload uniform vals that never change
	glProgramUniform1fv(m_program, 1, 1, &m_cfg.velocityColorLimit);
	glProgramUniform4fv(m_program, 2, 1, (GLfloat*)&m_cfg.velocityColor);
}

SimulationScene::~SimulationScene()
{
	glDeleteProgram(m_program);
	glDeleteBuffers(1, &m_vbo);
	glDeleteVertexArrays(1, &m_vao);
	glDeleteBuffers(1, &m_ssbo);

	cuCheck(cudaFree(m_deviceParticlesIn));
	cuCheck(cudaFree(m_deviceParticlesOut));
	cuCheck(cudaFree(m_deviceCollisions));
	free(m_collisions);
	free(m_newParticles);
}

void SimulationScene::spawnParticles(const pVec& pos, cudaStream_t stream)
{
	const unsigned int newParticleCount = std::min(m_numParticles + m_cfg.numParticlesToSpawn, m_cfg.maxNumParticles);
	const unsigned int numNewParticles = newParticleCount - m_numParticles;

	if (numNewParticles != 0)
	{
		std::uniform_real_distribution<float> redDis = std::uniform_real_distribution<float>(m_cfg.particleRedRange.x, m_cfg.particleRedRange.y);
		std::uniform_real_distribution<float> greenDis = std::uniform_real_distribution<float>(m_cfg.particleGreenRange.x, m_cfg.particleGreenRange.y);
		std::uniform_real_distribution<float> blueDis = std::uniform_real_distribution<float>(m_cfg.particleBlueRange.x, m_cfg.particleBlueRange.y);

		//spawn particles along the circumference of a circle
		if (numNewParticles == 1)
		{
			m_newParticles[0].position = pos;
			m_newParticles[0].velocity = pVec(0.0f);
			m_newParticles[0].color = glm::vec4(redDis(rng), greenDis(rng), blueDis(rng), 1.0f);
		}
		else
		{
			const float angleIncrement = 6.2832f / numNewParticles;
			const float circleRadius = numNewParticles * 2.0f;
			for (unsigned int i = 0; i < numNewParticles; i++)
			{
				m_newParticles[i].position = pos + pVec(cos(i * angleIncrement) * circleRadius, sin(i * angleIncrement) * circleRadius);
				m_newParticles[i].velocity = pVec(0.0f);
				m_newParticles[i].color = glm::vec4(redDis(rng), greenDis(rng), blueDis(rng), 1.0f);
			}
		}

		cudaMemcpyAsync(m_deviceParticlesIn + m_numParticles, m_newParticles, sizeof(Particle) * numNewParticles, cudaMemcpyHostToDevice, stream);
		m_numParticles = newParticleCount;
	}
}

void SimulationScene::update(float deltaTime)
{
	cudaStream_t stream;
	cuCheck(cudaStreamCreate(&stream));

	pVec pos = getCursorPos();

	//spawn particles into scene based on user input
	bool spawn = _window->mousePressed(GLFW_MOUSE_BUTTON_3);
	if (spawn) spawnParticles(pos, stream);

	dim3 block1D(ceil((float)m_numParticles / (float)m_maxNumThreads));
	dim3 thread1D(std::min(m_numParticles, m_maxNumThreads));

	dim3 block2D(m_numParticles);
	dim3 thread2D(std::min(m_numParticles, m_maxNumThreads));
	
	//apply explode or suction based on user input (reads from particlesIn, stores results in particlesIn)
	bool explode = _window->mousePressed(GLFW_MOUSE_BUTTON_1);
	bool suction = _window->mousePressed(GLFW_MOUSE_BUTTON_2);
	if (explode)      cudaExplode<<<block1D, thread1D, 0, stream>>>(m_deviceParticlesIn, m_numParticles, deltaTime, pos);
	else if (suction) cudaSuction<<<block1D, thread1D, 0, stream>>>(m_deviceParticlesIn, m_numParticles, deltaTime, pos);

	//main physics computation; reads from particlesIn, stores results in particlesOut
	cudaRun<<<block2D,thread2D,0,stream>>>(m_deviceParticlesIn, m_deviceParticlesOut, m_deviceCollisions, m_numParticles, deltaTime);

	//map opengl particles into memory; reads from particlesOut, stores results in the SSBO
	Particle* particles;
	size_t size;
	cuCheck(cudaGraphicsMapResources(1, &m_ssboResource, stream));
	cuCheck(cudaGraphicsResourceGetMappedPointer((void**)&particles, &size, m_ssboResource));
	cuCheck(cudaMemcpyAsync(particles, m_deviceParticlesOut, sizeof(Particle) * m_numParticles, cudaMemcpyDeviceToDevice, stream));
	cuCheck(cudaGraphicsUnmapResources(1, &m_ssboResource, stream));

	//particlesOut will become particlesIn for the next iteration
	swapDeviceParticles();

	//we don't have access to the window resize callback,
	//so always recalculate view_proj in case a window resize happened
	glm::ivec2 frameBufferSize = _window->getFramebufferSize();
	glm::mat4 view_proj = glm::ortho((float)-frameBufferSize.x / 2, (float)frameBufferSize.x / 2, (float)-frameBufferSize.y / 2, (float)frameBufferSize.y / 2);
	glUniformMatrix4fv(0, 1, false, (GLfloat*)&view_proj);
	
	cuCheck(cudaStreamSynchronize(stream));
	cuCheck(cudaStreamDestroy(stream));

	//if audio output is enabled and sfx were loaded in
	if (m_cfg.audioOn && m_sounds.size() > 0)
	{
		//get collision data, find highest velocity "force" from this iteration's collisions
		cuCheck(cudaMemcpy(m_collisions, m_deviceCollisions, m_numParticles * sizeof(Collision), cudaMemcpyDeviceToHost));
		float maxForce = 0.0f;
		for (unsigned int i = 0; i < m_numParticles; i++)
		{
			if (m_collisions[i].force > maxForce) maxForce = m_collisions[i].force;
		}

		//if the highest force is sufficiently high, play a sound effect
		if (maxForce > 0.5f)
		{
			//get random sound; if it is currently playing, search all sounds until a stopped one is found
			std::uniform_int_distribution<unsigned int> soundDis = std::uniform_int_distribution<unsigned int>(0, m_sounds.size() - 1);
			unsigned int index = soundDis(rng);
			unsigned int startIndex = index;
			SoundPlayer* sound;
			do
			{
				sound = m_sounds[index];
				index = (index + 1) % m_sounds.size();
			} while (sound->isPlaying() == true && index != startIndex);

			//if a stopped sound was available, play it
			if (index != startIndex)
			{
				//biggest force = louder and lower pitch
				sound->setVolume(maxForce);
				sound->setPitch(lerp(m_cfg.maxPitch, m_cfg.minPitch, std::min(maxForce / m_cfg.minPitchForce, 1.0f)));
				sound->play();
			}
		}
	}
}

void SimulationScene::render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, m_cfg.displayParticleVertices, m_numParticles);
}

void SimulationScene::switchFrom(const std::string& previousScene, void* data)
{
	glBindVertexArray(m_vao);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
	glUseProgram(m_program);
}
