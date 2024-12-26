#include "filter/particle_filter.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

int main() {
    // Simulation parameters
    const size_t num_particles = 100;
    const size_t top_percentage = 25;
    const float noise_x = 2.0, noise_y = 2.0;
    const float noise_vx = 0.1, noise_vy = 0.1;
    const float noise_w = 0.05, noise_h = 0.05;
    const float noise_qw = 0.01, noise_qx = 0.01, noise_qy = 0.01, noise_qz = 0.01;

    // Initialize ParticleFilter
    particle_filter::ParticleFilter tracker(num_particles, noise_x, noise_y, 
                                             noise_vx, noise_vy, 
                                             noise_w, noise_h, 
                                             noise_qw, noise_qx, noise_qy, noise_qz, 
                                             top_percentage);

    // Random number generator for initialization
    std::mt19937 rng(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> pos_dist(0.0, 100.0);
    std::uniform_real_distribution<float> vel_dist(-1.0, 1.0);
    std::uniform_real_distribution<float> size_dist(50.0, 150.0);

    // Define target states for objects
    struct TargetState {
        float x, y, vx, vy, w, h, qw, qx, qy, qz;
    };
    std::unordered_map<int, TargetState> targets;

    // Add objects to track and initialize target states
    for (int i = 1; i <= 3; ++i) 
    {
        float init_x = pos_dist(rng);
        float init_y = pos_dist(rng);
        float init_vx = vel_dist(rng);
        float init_vy = vel_dist(rng);
        float init_w = size_dist(rng);
        float init_h = size_dist(rng);
        float init_qw = 1.0;
        float init_qx = 0.0;
        float init_qy = 0.0;
        float init_qz = 0.0;

        tracker.initialize(i, init_x, init_y, init_vx, init_vy, init_w, init_h, init_qw, init_qx, init_qy, init_qz);

        targets[i] = {init_x, init_y, init_vx, init_vy, init_w, init_h, init_qw, init_qx, init_qy, init_qz};
    }

    // Simulate updates for 20 steps
    for (int step = 0; step < 20; ++step) 
    {
        std::cout << "Step " << step + 1 << ":" << std::endl;

        // Evolve target states (simple linear motion)
        for (auto& [id, target] : targets) 
        {
            target.x += target.vx;
            target.y += target.vy;
        }

        // Predict and update each object separately
        std::normal_distribution<float> noise_dist(0.0, 1.0); // Gaussian noise
        for (const auto& [id, target] : targets) 
        {
            // Predict step for the current object
            tracker.predict(id);

            // Generate simulated observations with noise
            float obs_x = target.x + noise_dist(rng);
            float obs_y = target.y + noise_dist(rng);
            float obs_vx = target.vx + noise_dist(rng) * 0.1f;
            float obs_vy = target.vy + noise_dist(rng) * 0.1f;
            float obs_w = target.w + noise_dist(rng) * 0.5f;
            float obs_h = target.h + noise_dist(rng) * 0.5f;
            float obs_qw = target.qw; // Assuming perfect quaternion observation
            float obs_qx = target.qx;
            float obs_qy = target.qy;
            float obs_qz = target.qz;

            // Update step for the current object
            tracker.updateObject(id, obs_x, obs_y, obs_vx, obs_vy, obs_w, obs_h, obs_qw, obs_qx, obs_qy, obs_qz);
        }

        // Print estimated states and calculate errors
        for (const auto& [id, target] : targets) 
        {
            if (tracker.isTrackingObject(id)) 
            {
                std::vector<float> state = tracker.getObjectState(id);
                float error_x = std::abs(state[0] - target.x);
                float error_y = std::abs(state[1] - target.y);
                float error_vx = std::abs(state[2] - target.vx);
                float error_vy = std::abs(state[3] - target.vy);

                std::cout << "Object " << id << " state: "
                          << "x = " << state[0] << ", y = " << state[1]
                          << ", vx = " << state[2] << ", vy = " << state[3]
                          << ", w = " << state[4] << ", h = " << state[5]
                          << ", qw = " << state[6] << ", qx = " << state[7]
                          << ", qy = " << state[8] << ", qz = " << state[9]
                          << std::endl;

                std::cout << "Target state: "
                          << "x = " << target.x << ", y = " << target.y
                          << ", vx = " << target.vx << ", vy = " << target.vy
                          << std::endl;

                std::cout << "Errors: "
                          << "x = " << error_x << ", y = " << error_y
                          << ", vx = " << error_vx << ", vy = " << error_vy
                          << std::endl;
            }
        }

        std::cout << "----------------------------------" << std::endl;
    }

    return 0;
}
