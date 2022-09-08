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

//number of particles in the simulation
const unsigned int SimulationScene::maxNumParticles = 5120;

//number of particles to spawn when user clicks on screen
const unsigned int SimulationScene::numParticlesToSpawn = 5;

//number of particles which should already be in scene at start
const unsigned int SimulationScene::numParticlesAtStart = 0;

//radius of the particle to render
const float SimulationScene::displayParticleRadius = 5.0f;

//number of vertices on the circumference of a particle's circle (-2 to include origin and first vertex twice)
const unsigned int SimulationScene::displayParticleVertices = 12;

//color value ranges for the particles
const glm::vec2 SimulationScene::particleRedRange = glm::vec2(0.0f, 0.05f);
const glm::vec2 SimulationScene::particleGreenRange = glm::vec2(0.0f, 0.1f);
const glm::vec2 SimulationScene::particleBlueRange = glm::vec2(0.65f, 1.0f);

//radius of the particle in the simulation
const float SimulationScene::particleRadius = 5.0f;

//how much to multiply velocity by each frame
const SimulationScene::pVec SimulationScene::velocityDullingFactor = SimulationScene::pVec(0.98f, 1.0f);

//velocity will be dulled this many times a second
const float SimulationScene::velocityDullingFactorRate = 60.0f;

//how elastic/inelastic the particle collisions are; 0.0f = inelastic, 1.0f = elastic
const float SimulationScene::restitutionCoefficient = 0.2f;

//constant force applied to simulation
const SimulationScene::pVec SimulationScene::gravity = SimulationScene::pVec(0.0f, -9.81f);

//bottom left corner of simulation
const SimulationScene::pVec SimulationScene::boundMin = SimulationScene::pVec(-250.0f, -250.0f);

//top right corner of simulation
const SimulationScene::pVec SimulationScene::boundMax = SimulationScene::pVec(250.0f, 250.0f);

//distance where a particle will be impacted by an explosion
const float SimulationScene::maxExplodeRange = 50.0f;

//the velocity of a particle right next to an explosion (tapers off as it is farther away)
const float SimulationScene::maxExplodeForce = 20.0f;

//distance where a particle will be impacted by suction
const float SimulationScene::maxSuctionRange = 100.0f;

//the velocity of a particle right next to the suction (tapers off as it is farther away)
const float SimulationScene::maxSuctionForce = 10.0f;

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
	const unsigned int numParticles,
	const float deltaTime)
{
	if (threadIdx.x == 0) { d_velocityDelta = SimulationScene::pVec(0.0f); d_positionDelta = SimulationScene::pVec(0.0f); }
	__syncthreads();

	const unsigned int index = blockIdx.x;
	const SimulationScene::Particle in = particlesIn[index];
	SimulationScene::Particle* const out = particlesOut + index;

	const SimulationScene::pVec inVelocity = in.velocity;
	const SimulationScene::pVec inPosition = in.position + inVelocity * deltaTime;

	//get particles to check to check for collisions
	const unsigned int numOtherParticles = cudaCeil((float)numParticles / (float)blockDim.x);
	const unsigned int firstOtherParticle = threadIdx.x * numOtherParticles;
	const unsigned int lastOtherParticle = cudaMin(firstOtherParticle + numOtherParticles, numParticles);

	//compute velocity based on colliding particles
	SimulationScene::pVec velocityDelta = SimulationScene::pVec(0.0f);
	SimulationScene::pVec positionDelta = SimulationScene::pVec(0.0f);

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
				velocityDelta += d_restitutionCoefficient * -cudaDotProduct(inVelocity - otherVelocity, delta) * delta / distanceSqr;
			}
		}
	}
	#pragma unroll
	for (char i = 0; i < PVEC_DIM; i++)
	{
		atomicAdd(&d_velocityDelta[i], velocityDelta[i]);
		atomicAdd(&d_positionDelta[i], positionDelta[i]);
	}
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
			}
			else if (outPosition[i] > d_boundMax[i])
			{
				outPosition[i] = d_boundMax[i];
				outVelocity[i] *= -d_restitutionCoefficient;
			}
		}

		//finally store results
		out->position = outPosition;
		out->velocity = outVelocity;
		out->color = in.color;
	}
}

__device__ float cudaLerp(const float x0, const float x1, const float t)
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

SimulationScene::SimulationScene()
	: Scene("Simulation")
{
	//generate distributions based on size of simulation bounding box, expected color values
	std::uniform_real_distribution<float> dis[PVEC_DIM];
	for (char i = 0; i < PVEC_DIM; i++)
	{
		dis[i] = std::uniform_real_distribution<float>(boundMin[i], boundMax[i]);
	}
	std::uniform_real_distribution<float> redDis = std::uniform_real_distribution<float>(particleRedRange.x, particleRedRange.y);
	std::uniform_real_distribution<float> greenDis = std::uniform_real_distribution<float>(particleGreenRange.x, particleGreenRange.y);
	std::uniform_real_distribution<float> blueDis = std::uniform_real_distribution<float>(particleBlueRange.x, particleBlueRange.y);

	m_numParticles = numParticlesAtStart;

	Particle* particles = (Particle*)malloc(maxNumParticles * sizeof(Particle));

	//initialize random positions and colors
	for (unsigned int i = 0; i < m_numParticles; i++)
	{ 
		for (char j = 0; j < PVEC_DIM; j++)
		{
			particles[i].position[j] = dis[j](rng);
		}
		particles[i].velocity = pVec(0.0f);
		particles[i].color = glm::vec4(redDis(rng), greenDis(rng), blueDis(rng), 1.0f);
	};

	//put the particle data into a SSBO
	glGenBuffers(1, &m_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, maxNumParticles * sizeof(Particle), particles, GL_DYNAMIC_COPY);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//generate triangle fan of circle vertices
	glm::vec2* circlePoints = (glm::vec2*)malloc(displayParticleVertices * sizeof(glm::vec2));
	float angleInterval = 6.283185f / (displayParticleVertices - 2);
	circlePoints[0] = glm::vec2(0.0f, 0.0f);
	for (unsigned int i = 1; i < displayParticleVertices; i++)
	{
		const float angle = (i - 1) * angleInterval;
		circlePoints[i] = glm::vec2(cos(angle) * displayParticleRadius, sin(angle) * displayParticleRadius);
	}

	//VAO with 1 VBOs for the circle
	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);
	glGenBuffers(1, &m_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glBufferData(GL_ARRAY_BUFFER, displayParticleVertices * sizeof(glm::vec2), circlePoints, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	free(circlePoints);

	//cuda can write to the particle vertices via m_ssboResource
	cuCheck(cudaGraphicsGLRegisterBuffer(&m_ssboResource, m_ssbo, cudaGraphicsRegisterFlagsWriteDiscard));

	//allocate memory for cuda particles, copy data into the first one
	cuCheck(cudaMalloc((void**)&m_deviceParticlesIn, maxNumParticles * sizeof(Particle)));
	cuCheck(cudaMalloc((void**)&m_deviceParticlesOut, maxNumParticles * sizeof(Particle)));
	cuCheck(cudaMemcpy(m_deviceParticlesIn, particles, maxNumParticles * sizeof(Particle), cudaMemcpyHostToDevice));
	free(particles);

	//copy simulation constants into constant cuda memory
	cuCheck(cudaMemcpyToSymbol(d_particleRadius, &particleRadius, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_velocityDullingFactor, &velocityDullingFactor, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_velocityDullingFactorRate, &velocityDullingFactorRate, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_restitutionCoefficient, &restitutionCoefficient, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_gravity, &gravity, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_boundMin, &boundMin, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_boundMax, &boundMax, sizeof(pVec), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxExplodeRange, &maxExplodeRange, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxExplodeForce, &maxExplodeForce, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxSuctionRange, &maxSuctionRange, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuCheck(cudaMemcpyToSymbol(d_maxSuctionForce, &maxSuctionForce, sizeof(float), 0, cudaMemcpyHostToDevice));

	//get properties from user's GPU to ensure maximum performance of simulation
	cudaDeviceProp properties;
	cuCheck(cudaGetDeviceProperties(&properties, 0));
	m_maxNumBlocks = properties.maxGridSize[0];
	m_maxNumThreads = properties.maxThreadsDim[0];
	if (m_maxNumBlocks < maxNumParticles)
	{
		fprintf(stderr, "Error: gpu's max grid size is %d, which is too low for %d particles", m_maxNumBlocks, maxNumParticles);
		abort();
	}

	//create rendering program from shaders
	m_program = _util->createProgramFromDisk("shaders\\Vert.shader", "shaders\\Frag.shader");
}

SimulationScene::~SimulationScene()
{
	glDeleteProgram(m_program);
	glDeleteBuffers(1, &m_vbo);
	glDeleteVertexArrays(1, &m_vao);
	glDeleteBuffers(1, &m_ssbo);

	cuCheck(cudaFree(m_deviceParticlesIn));
	cuCheck(cudaFree(m_deviceParticlesOut));
}

void SimulationScene::spawnParticles(const pVec& pos, cudaStream_t stream)
{
	Particle newParticles[numParticlesToSpawn];
	const unsigned int newParticleCount = std::min(m_numParticles + numParticlesToSpawn, maxNumParticles);
	const unsigned int numNewParticles = newParticleCount - m_numParticles;

	if (numNewParticles != 0)
	{
		std::uniform_real_distribution<float> redDis = std::uniform_real_distribution<float>(particleRedRange.x, particleRedRange.y);
		std::uniform_real_distribution<float> greenDis = std::uniform_real_distribution<float>(particleGreenRange.x, particleGreenRange.y);
		std::uniform_real_distribution<float> blueDis = std::uniform_real_distribution<float>(particleBlueRange.x, particleBlueRange.y);

		//spawn particles along the circumference of a circle
		if (numNewParticles == 1)
		{
			newParticles[0].position = pos;
			newParticles[0].velocity = pVec(0.0f);
		}
		else
		{
			const float angleIncrement = 6.2832f / numNewParticles;
			const float circleRadius = numNewParticles * 2.0f;
			for (unsigned int i = 0; i < numNewParticles; i++)
			{
				newParticles[i].position = pos + pVec(cos(i * angleIncrement) * circleRadius, sin(i * angleIncrement) * circleRadius);
				newParticles[i].velocity = pVec(0.0f);
				newParticles[i].color = glm::vec4(redDis(rng), greenDis(rng), blueDis(rng), 1.0f);
			}
		}

		cudaMemcpyAsync(m_deviceParticlesIn + m_numParticles, newParticles, sizeof(Particle) * numNewParticles, cudaMemcpyHostToDevice, stream);
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
	cudaRun<<<block2D,thread2D,0,stream>>>(m_deviceParticlesIn, m_deviceParticlesOut, m_numParticles, deltaTime);

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
}

void SimulationScene::render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, displayParticleVertices, m_numParticles);
}

void SimulationScene::switchFrom(const std::string& previousScene, void* data)
{
	glBindVertexArray(m_vao);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_ssbo);
	glUseProgram(m_program);
}
