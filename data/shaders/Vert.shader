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

#version 430 core

layout(location = 0) in vec2 inPosition;

struct Particle
{
	vec2 position;
	vec2 velocity;
	vec4 color;
};

layout(std140, binding = 0) readonly buffer Particles
{
	Particle particles[2048];
};

layout(location = 0) uniform mat4 view_proj;

out vec4 col;

float lerp(float x0, float x1, float t)
{
	return x0 + t * (x1 - x0);
}

void main()
{
	Particle p = particles[gl_InstanceID];
	gl_Position = view_proj * vec4(inPosition.x + p.position.x, inPosition.y + p.position.y, 0.0f, 1.0f);

	const float velocityMag = sqrt(dot(p.velocity, p.velocity));
	const float maxVelocity = 100.0f;
	float velocityColor = lerp(0.0f, 1.0f, clamp(velocityMag / maxVelocity, 0.0f, 1.0f));
	col = p.color + velocityColor;
}
