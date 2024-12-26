#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstddef>

namespace particle_filter
{
    class ParticleStates 
    {
        public:
            /// @brief Constructor: Initialize device vectors with number of particles
            explicit ParticleStates(size_t num_particles);

            /// @brief Returns the size of each state, i.e num of particles
            size_t size();

            /// @brief Number of states
            size_t size_states();

            /// @brief Move GPU data to CPU data
            void download();

            /// @brief Move CPU data to GPU data 
            void upload();

            /// @brief Get raw pointers to device vectors
            float* device_x(); 
            float* device_y(); 
            float* device_vx();
            float* device_vy();
            float* device_w();
            float* device_h();
            float* device_qw();
            float* device_qx();
            float* device_qy();
            float* device_qz();

            /// @brief Get raw pointers to host vectors
            float* host_x();
            float* host_y();
            float* host_vx();
            float* host_vy();
            float* host_w();
            float* host_h();
            float* host_qw();
            float* host_qx();
            float* host_qy();
            float* host_qz();

            /// @brief Get raw pointers to mean, cov, weights vector GPU
            float* device_mean();
            float* device_cov();
            float* device_weights();

            /// @brief Get raw pointers to mean, cov, weights vector CPU
            float* host_mean();
            float* host_cov();
            float* host_weights();

        private:
            size_t num_particles_; // Total number of particles

            void resize_(size_t num_particles);

            // Device vectors (GPU)
            thrust::device_vector<float> d_x_;
            thrust::device_vector<float> d_y_;
            thrust::device_vector<float> d_vx_;
            thrust::device_vector<float> d_vy_;
            thrust::device_vector<float> d_w_;
            thrust::device_vector<float> d_h_;
            thrust::device_vector<float> d_qw_;
            thrust::device_vector<float> d_qx_;
            thrust::device_vector<float> d_qy_;
            thrust::device_vector<float> d_qz_;

            // Host vectors (CPU)
            thrust::host_vector<float> h_x_;
            thrust::host_vector<float> h_y_;
            thrust::host_vector<float> h_vx_;
            thrust::host_vector<float> h_vy_;
            thrust::host_vector<float> h_w_;
            thrust::host_vector<float> h_h_;
            thrust::host_vector<float> h_qw_;
            thrust::host_vector<float> h_qx_;
            thrust::host_vector<float> h_qy_;
            thrust::host_vector<float> h_qz_;

            // Mean, Cov, weights Device (GPU)
            thrust::device_vector<float> d_mean_;
            thrust::device_vector<float> d_cov_;
            thrust::device_vector<float> d_weights_;

            // Mean, Cov, weights Host (CPU)
            thrust::host_vector<float> h_mean_;
            thrust::host_vector<float> h_cov_;
            thrust::host_vector<float> h_weights_;

    };

} // namespace particle_filter
