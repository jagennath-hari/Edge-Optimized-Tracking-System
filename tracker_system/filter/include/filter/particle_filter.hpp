#pragma once

#include <chrono>
#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include "filter/particle_states.cuh"
#include "filter/kernels.cuh"
#include "filter_interface.hpp"

namespace particle_filter
{
    class ParticleFilter : public IFilter
    {
    public:
        /// @brief Class constructor
        ParticleFilter(size_t num_particles, const float noise_x, const float noise_y, 
            const float noise_vx, const float noise_vy, 
            const float noise_w, const float noise_h, 
            const float noise_qw, const float noise_qx, const float noise_qy, const float noise_qz, 
            size_t top_percentage = 25
        );

        /// @brief Add a new object to track
        void initialize(int object_id, 
                    float init_x, float init_y, float init_vx, float init_vy, 
                    float init_w, float init_h, 
                    float init_qw, float init_qx, float init_qy, float init_qz) override;
        
        /// @brief Remove an object from tracking
        void removeObject(int object_id) override;

        /// @brief Predict step for the object ID
        void predict(int object_id) override;

        /// @brief Update step for a single object
        void updateObject(int object_id, 
                        float obs_x, float obs_y, float obs_vx, float obs_vy, 
                        float obs_w, float obs_h, 
                        float obs_qw, float obs_qx, float obs_qy, float obs_qz) override;

        /// @brief Get the state of an object (mean of top percentage particles)
        std::vector<float> getObjectState(int object_id) const override;

        /// @brief Get all particles for an object
        std::vector<std::vector<float>> getObjectParticles(int object_id) const override;

        /// @brief Check if an object is being tracked
        bool isTrackingObject(int object_id) const override;

    private:
        size_t num_particles_;  // Number of particles per object
        size_t top_percentage_; // Top percentage of particles for state estimation

        // Struct for object state
        struct ObjectState {
            std::unique_ptr<particle_filter::ParticleStates> particles;
        };

        // Map of object ID to their particle state
        std::unordered_map<int, ObjectState> objects_;

        // Helper function to compute mean state of top-weighted particles
        std::vector<float> computeMeanState_(particle_filter::ParticleStates& particles) const;

        // Noise values
        const float noise_x_;
        const float noise_y_; 
        const float noise_vx_;
        const float noise_vy_; 
        const float noise_w_;
        const float noise_h_; 
        const float noise_qw_;
        const float noise_qx_;
        const float noise_qy_;
        const float noise_qz_;
    };
}