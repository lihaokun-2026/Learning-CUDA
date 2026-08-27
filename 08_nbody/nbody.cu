#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t error__ = (call);                                          \
        if (error__ != cudaSuccess) {                                          \
            std::ostringstream message__;                                      \
            message__ << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(error__);                 \
            throw std::runtime_error(message__.str());                         \
        }                                                                       \
    } while (false)

constexpr int TILE_SIZE = 256;

struct Params {
    float dt = 1.0e-3f;
    int steps = 1000;
    int record_interval = 100;
    float G = 1.0f;
    float softening = 1.0e-4f;
    std::string integrator = "leapfrog";
};

struct Body {
    float x, y, z, vx, vy, vz, mass;
};

__global__ void acceleration_kernel(const float4* positions, float4* acceleration,
                                    int n, float gravitational_constant,
                                    float softening_squared) {
    __shared__ float4 tile[TILE_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const float4 pi = active ? positions[i] : make_float4(0, 0, 0, 0);
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int base = 0; base < n; base += TILE_SIZE) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? positions[j] : make_float4(0, 0, 0, 0);
        __syncthreads();
        const int count = min(TILE_SIZE, n - base);
        if (active) {
            for (int k = 0; k < count; ++k) {
                const float4 pj = tile[k];
                const float dx = pj.x - pi.x;
                const float dy = pj.y - pi.y;
                const float dz = pj.z - pi.z;
                const float distance_squared = dx * dx + dy * dy + dz * dz + softening_squared;
                const float inv_distance = rsqrtf(distance_squared);
                const float scale = gravitational_constant * pj.w * inv_distance * inv_distance * inv_distance;
                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }
        __syncthreads();
    }
    if (active) acceleration[i] = make_float4(ax, ay, az, 0.0f);
}

__global__ void euler_kernel(float4* positions, float4* velocities,
                             const float4* acceleration, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 p = positions[i], v = velocities[i], a = acceleration[i];
    v.x += a.x * dt; v.y += a.y * dt; v.z += a.z * dt;
    p.x += v.x * dt; p.y += v.y * dt; p.z += v.z * dt;
    velocities[i] = v; positions[i] = p;
}

__global__ void kick_kernel(float4* velocities, const float4* acceleration, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    velocities[i].x += acceleration[i].x * dt;
    velocities[i].y += acceleration[i].y * dt;
    velocities[i].z += acceleration[i].z * dt;
}

__global__ void drift_kernel(float4* positions, const float4* velocities, int n, float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    positions[i].x += velocities[i].x * dt;
    positions[i].y += velocities[i].y * dt;
    positions[i].z += velocities[i].z * dt;
}

static void read_particles(const std::string& path, std::vector<Body>& bodies) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open particle file: " + path);
    Body body{};
    while (input >> body.x >> body.y >> body.z >> body.vx >> body.vy >> body.vz >> body.mass)
        bodies.push_back(body);
    if (bodies.empty() || !input.eof()) throw std::runtime_error("invalid particle file: " + path);
}

static void read_params(const std::string& path, Params& p) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open parameter file: " + path);
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.resize(comment);
        std::istringstream row(line);
        std::string key, equals, value;
        if (!(row >> key)) continue;
        if (key == "dt") row >> equals >> p.dt;
        else if (key == "num_steps") row >> equals >> p.steps;
        else if (key == "record_interval") row >> equals >> p.record_interval;
        else if (key == "G") row >> equals >> p.G;
        else if (key == "softening") row >> equals >> p.softening;
        else if (key == "integrator") row >> equals >> p.integrator;
    }
    if (p.dt <= 0 || p.steps < 0 || p.record_interval <= 0 || p.softening < 0)
        throw std::runtime_error("invalid simulation parameters");
    if (p.integrator != "euler" && p.integrator != "leapfrog")
        throw std::runtime_error("integrator must be euler or leapfrog");
}

static void write_trajectory(const std::string& path, int n, int records,
                             const std::vector<float>& trajectory) {
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create output file: " + path);
    const std::int32_t particle_count = n, record_count = records;
    output.write(reinterpret_cast<const char*>(&particle_count), sizeof(particle_count));
    output.write(reinterpret_cast<const char*>(&record_count), sizeof(record_count));
    output.write(reinterpret_cast<const char*>(trajectory.data()),
                 static_cast<std::streamsize>(trajectory.size() * sizeof(float)));
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: nbody <particles.txt> <params.txt> [trajectory.bin]\n";
        return 2;
    }
    try {
        std::vector<Body> bodies;
        Params params;
        read_particles(argv[1], bodies);
        read_params(argv[2], params);
        const std::string output_path = argc == 4 ? argv[3] : "trajectory.bin";
        const int n = static_cast<int>(bodies.size());
        const int records = params.steps / params.record_interval + 1;
        std::vector<float> trajectory(static_cast<size_t>(records) * n * 3);
        std::vector<float4> host_positions(n), host_velocities(n);
        for (int i = 0; i < n; ++i) {
            host_positions[i] = make_float4(bodies[i].x, bodies[i].y, bodies[i].z, bodies[i].mass);
            host_velocities[i] = make_float4(bodies[i].vx, bodies[i].vy, bodies[i].vz, 0);
        }
        float4 *positions = nullptr, *velocities = nullptr, *acceleration = nullptr;
        CUDA_CHECK(cudaMalloc(&positions, n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&velocities, n * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&acceleration, n * sizeof(float4)));
        CUDA_CHECK(cudaMemcpy(positions, host_positions.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(velocities, host_velocities.data(), n * sizeof(float4), cudaMemcpyHostToDevice));
        const int blocks = (n + TILE_SIZE - 1) / TILE_SIZE;
        auto record = [&](int index) {
            CUDA_CHECK(cudaMemcpy(host_positions.data(), positions, n * sizeof(float4), cudaMemcpyDeviceToHost));
            for (int i = 0; i < n; ++i) {
                trajectory[(static_cast<size_t>(index) * n + i) * 3 + 0] = host_positions[i].x;
                trajectory[(static_cast<size_t>(index) * n + i) * 3 + 1] = host_positions[i].y;
                trajectory[(static_cast<size_t>(index) * n + i) * 3 + 2] = host_positions[i].z;
            }
        };
        record(0);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto start = std::chrono::steady_clock::now();
        int record_index = 1;
        if (params.integrator == "leapfrog") {
            acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
            CUDA_CHECK(cudaGetLastError());
            kick_kernel<<<blocks, TILE_SIZE>>>(velocities, acceleration, n, 0.5f * params.dt);
            CUDA_CHECK(cudaGetLastError());
        }
        for (int step = 1; step <= params.steps; ++step) {
            if (params.integrator == "euler") {
                acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
                euler_kernel<<<blocks, TILE_SIZE>>>(positions, velocities, acceleration, n, params.dt);
            } else {
                drift_kernel<<<blocks, TILE_SIZE>>>(positions, velocities, n, params.dt);
                acceleration_kernel<<<blocks, TILE_SIZE>>>(positions, acceleration, n, params.G, params.softening * params.softening);
                kick_kernel<<<blocks, TILE_SIZE>>>(velocities, acceleration, n, 0.5f * params.dt);
            }
            CUDA_CHECK(cudaGetLastError());
            if (step % params.record_interval == 0) record(record_index++);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        write_trajectory(output_path, n, records, trajectory);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const double interactions = static_cast<double>(n) * n * params.steps;
        std::cout << "particles=" << n << " steps=" << params.steps << " records=" << records << "\n"
                  << "integrator=" << params.integrator << " elapsed_sec=" << std::fixed << std::setprecision(6) << seconds
                  << " particle_steps_per_sec=" << (seconds > 0 ? n * params.steps / seconds : 0) << "\n"
                  << "interactions_per_sec=" << (seconds > 0 ? interactions / seconds : 0) << "\n";
        cudaFree(positions); cudaFree(velocities); cudaFree(acceleration);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
