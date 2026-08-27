"""Generate deterministic N-body input cases for benchmarking and validation.

Usage:
    python generate_cases.py

The generated particle files use: x y z vx vy vz mass.
"""
from pathlib import Path
import math
import random

ROOT = Path(__file__).parent


def write_case(name, particles, params):
    (ROOT / f"{name}_particles.txt").write_text(
        "\n".join("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}".format(*p) for p in particles) + "\n",
        encoding="utf-8",
    )


def write_large_disk(name, count, params, seed):
    """Stream a large rotating 3D disk to avoid building a large Python list."""
    rng = random.Random(seed)
    path = ROOT / f"{name}_particles.txt"
    with path.open("w", encoding="utf-8") as output:
        for _ in range(count):
            radius = 0.5 + 9.5 * math.sqrt(rng.random())
            angle = 2 * math.pi * rng.random()
            z = rng.gauss(0, 0.03)
            speed = math.sqrt(1.0 / radius)
            values = (radius * math.cos(angle), radius * math.sin(angle), z,
                      -speed * math.sin(angle), speed * math.cos(angle),
                      rng.gauss(0, 0.002), 1.0 / count)
            output.write("{:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(*values))
    (ROOT / f"{name}_params.txt").write_text(
        "\n".join(f"{key} = {value}" for key, value in params.items()) + "\n",
        encoding="utf-8",
    )
    (ROOT / f"{name}_params.txt").write_text(
        "\n".join(f"{key} = {value}" for key, value in params.items()) + "\n",
        encoding="utf-8",
    )


def two_body():
    # Equal masses in a circular orbit around their center of mass.
    speed = math.sqrt(0.5)
    return [(-0.5, 0, 0, 0, -speed, 0, 1), (0.5, 0, 0, 0, speed, 0, 1)]


def solar_system():
    # Normalized Sun + four planets with different orbital inclinations.
    particles = [(0, 0, 0, 0, 0, 0, 1000.0)]
    planets = [(0.4, 0.01, -8), (0.7, 0.02, 4), (1.0, 0.03, 12), (1.5, 0.04, -16)]
    for radius, mass, inclination_degrees in planets:
        inclination = math.radians(inclination_degrees)
        speed = math.sqrt(1000.0 / radius)
        particles.append((radius, 0, 0, 0, speed * math.cos(inclination),
                          speed * math.sin(inclination), mass))
    return particles


def cluster(count=64, seed=7):
    rng = random.Random(seed)
    particles = []
    for _ in range(count):
        # Gaussian compact cluster with small random velocities.
        x, y, z = (rng.gauss(0, 1) for _ in range(3))
        vx, vy, vz = (rng.gauss(0, 0.08) for _ in range(3))
        particles.append((x, y, z, vx, vy, vz, 1.0 / count))
    return particles


def plummer_sphere(count=256, seed=23):
    # Isotropic 3D cloud with tangential swirl for an obvious 3D animation.
    rng = random.Random(seed)
    particles = []
    for _ in range(count):
        u = max(rng.random(), 1e-6)
        radius = min((u ** (-2.0 / 3.0) - 1.0) ** -0.5, 8.0)
        cos_theta = 2 * rng.random() - 1
        sin_theta = math.sqrt(1 - cos_theta * cos_theta)
        phi = 2 * math.pi * rng.random()
        x = radius * sin_theta * math.cos(phi)
        y = radius * sin_theta * math.sin(phi)
        z = radius * cos_theta
        vx = -0.12 * y + rng.gauss(0, 0.025)
        vy = 0.12 * x + rng.gauss(0, 0.025)
        vz = rng.gauss(0, 0.04)
        particles.append((x, y, z, vx, vy, vz, 1.0 / count))
    return particles


def disk(count=1024, seed=11):
    rng = random.Random(seed)
    particles = [(0, 0, 0, 0, 0, 0, 1000.0)]
    for _ in range(count):
        radius = 0.5 + 4.5 * math.sqrt(rng.random())
        angle = 2 * math.pi * rng.random()
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        velocity = math.sqrt(1000.0 / radius)
        particles.append((x, y, rng.gauss(0, 0.01), -velocity * math.sin(angle),
                          velocity * math.cos(angle), rng.gauss(0, 0.01), 1e-3))
    return particles


COMMON = {"dt": "1e-3", "record_interval": "100", "G": "1.0", "softening": "1e-4", "integrator": "leapfrog"}
write_case("two_body", two_body(), {**COMMON, "num_steps": "2000"})
write_case("solar_system", solar_system(), {**COMMON, "dt": "1e-4", "num_steps": "5000", "record_interval": "500"})
write_case("cluster_64", cluster(), {**COMMON, "dt": "2e-4", "num_steps": "2000"})
write_case("plummer_256", plummer_sphere(), {**COMMON, "dt": "2e-3", "num_steps": "4000", "record_interval": "40", "softening": "2e-2"})
write_case("disk_1024", disk(), {**COMMON, "dt": "1e-4", "num_steps": "1000", "record_interval": "100"})
write_large_disk("benchmark_4096", 4096,
                 {**COMMON, "dt": "1e-3", "num_steps": "1000", "record_interval": "100", "softening": "1e-2"}, 41)
write_large_disk("benchmark_65536", 65536,
                 {**COMMON, "dt": "1e-3", "num_steps": "1000", "record_interval": "100", "softening": "1e-2"}, 43)
print("Generated validation cases plus benchmark_4096 and benchmark_65536 in", ROOT)
