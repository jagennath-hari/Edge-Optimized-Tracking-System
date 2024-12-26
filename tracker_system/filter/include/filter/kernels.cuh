#pragma once

#include <chrono>
#include <cmath>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

namespace particle_filter_kernels
{
    #define REGULARIZATION 1e-6f

    /// @brief Kernel to initialize particles with a normal distribution
    __global__ void initialize_particles(
        float* x, float* y, float* vx, float* vy, float* w, float* h, float* qw, float* qx, float* qy, float* qz,
        float* weights,
        float initial_x, float initial_y, float initial_vx, float initial_vy, float initial_w, float initial_h, float initial_qw, float initial_qx, float initial_qy, float initial_qz,
        size_t num_particles,
        float stddev_x, float stddev_y, float stddev_vx, float stddev_vy, float stddev_w, float stddev_h, float stddev_qw, float stddev_qx, float stddev_qy, float stddev_qz,
        unsigned int seed
    );

    /// @brief Function to launch the intialize particles kernel
    void initParticles(
        float* d_x, float* d_y, float* d_vx, float* d_vy, float* d_w, float* d_h, float* d_qw, float* d_qx, float* d_qy, float* d_qz,
        float* d_weights,
        float initial_x, float initial_y, float initial_vx, float initial_vy, float initial_w, float initial_h, float initial_qw, float initial_qx, float initial_qy, float initial_qz,
        size_t num_particles,
        float stddev_x, float stddev_y, float stddev_vx, float stddev_vy, float stddev_w, float stddev_h, float stddev_qw, float stddev_qx, float stddev_qy, float stddev_qz,
        unsigned int seed = 12345
    );

    /// @brief Kernel to compute the mean
    __global__ void compute_mean(
        const float* x, const float* y, const float* vx, const float* vy,
        const float* w, const float* h, const float* qw, const float* qx,
        const float* qy, const float* qz, const float* weights,
        float* mean, size_t num_particles
    );

    /// @brief Function to launch the mean kernel
    void computeMean(
        const float* d_x, const float* d_y, const float* d_vx, const float* d_vy,
        const float* d_w, const float* d_h, const float* d_qw, const float* d_qx,
        const float* d_qy, const float* d_qz, const float* d_weights,
        float* d_mean, size_t num_particles
    );

    /// @brief Kernel to compute the covariance
    __global__ void compute_covariance(
        const float* x, const float* y, const float* vx, const float* vy,
        const float* w, const float* h, const float* qw, const float* qx,
        const float* qy, const float* qz, const float* weights,
        const float* mean, float* covariance, size_t num_particles
    );

    /// @brief Kernel to make covariance symmetric
    __global__ void symmetrize_covariance(
        float* covariance, size_t n
    );

    /// @brief Function to launch the covariance kernel
    void computeCovariance(
        const float* d_x, const float* d_y, const float* d_vx, const float* d_vy,
        const float* d_w, const float* d_h, const float* d_qw, const float* d_qx,
        const float* d_qy, const float* d_qz, const float* d_weights,
        const float* d_mean, float* d_covariance, size_t num_particles
    );

    /// @brief Kernel to compute the weights mean and covaraince of the sigma points
    __global__ void compute_weights_sigma(
        size_t n, float lambda, float alpha, float beta,
        float* weights_mean, float* weights_cov
    );

    /// @brief Function to compute the weights mean and covaraince of the sigma points by launching the kernel
    void computeWeightsMeanCovSigma(
        size_t n, float alpha, float beta, float kappa,
        float* d_weights_mean, float* d_weights_cov
    );

    /// @brief Function to Cholesky Decomposition using cuSOLVER
    void computeCholeskyDecomposition(
        const float* d_covariance, 
        float* d_cholesky,         
        size_t n                   
    );

    /// @brief Kernel to compute the sigma points
    __global__ void compute_sigma_points(
        const float* d_mean,
        const float* d_cholesky,
        float* d_sigma_points,
        size_t n
    );

    /// @brief Function to compute sigma points using the kernel
    void computeSigmaPoints(
        const float* d_mean,         
        const float* d_cholesky,    
        float* d_sigma_points,       
        size_t n                     
    );

    /// @brief Function to perform an unscented transform and returns a thurst tuple holding device vectors of the sigma points, weights mean, weights cov
    thrust::tuple<thrust::host_vector<float>, thrust::host_vector<float>, thrust::host_vector<float>> unscentedTransform(
        const float* d_mean,                             // Mean vector (n elements)
        const float* d_covariance,                       // Covariance matrix (n*n elements)
        float alpha,                                     // Scaling parameter alpha
        float beta,                                      // Scaling parameter beta
        float kappa                                      // Scaling parameter kappa
    );

    /// @brief Kernel to propogate sigma points
    __global__ void propagate_sigma_points(
        float* sigma_points,
        size_t num_sigma_points,
        size_t n,
        float dt,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz,
        unsigned int seed
    );

    /// @brief Function to launch the propogation of sigma points
    void propagateSigmaPoints(
        float* d_sigma_points,
        float dt,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz,
        unsigned int seed
    );

    /// @brief Kernel to compute the mean of propogated sigma points
    __global__ void compute_mean_propogated(
        float* sigma_points,
        float* weights_mean,
        float* predicted_mean,
        size_t num_sigma_points,
        size_t n
    );

    /// @brief Kernel to subbract the mean and compute the delta
    __global__ void subtract_mean(
        const float* sigma_points,   
        const float* predicted_mean, 
        float* delta_sigma_points,   
        size_t num_sigma_points,     
        size_t n                     
    );

    /// @brief Kernel to scale the delta sigma points
    __global__ void scale_delta_sigma_points(
        float* d_delta_sigma_points,
        const float* d_weights_cov,
        size_t num_sigma_points,
        size_t n
    );

    /// @brief Function to compute the Covaraince with cuBLAS
    void computeCovarianceWithCuBLAS(
        cublasHandle_t cublas_handle,
        const float* d_delta_sigma_points, // Input: Scaled deviations
        float* d_predicted_covariance,    // Output: Covariance matrix
        size_t num_sigma_points,
        size_t n                          // Number of states
    );

    /// @brief Kernel to compute the covaraince of propogated sigma points
    __global__ void compute_covariance_propogated(
        float* sigma_points,
        float* weights_cov,
        float* predicted_mean,
        float* predicted_covariance,
        size_t num_sigma_points,
        size_t n
    );

    /// @brief Function to launch the kernel to compute mean and covariance
    void computeMeanAndCovariance(
        float* d_sigma_points,
        float* d_weights_mean,
        float* d_weights_cov,
        float* d_predicted_mean,
        float* d_predicted_covariance,
        float* d_delta_sigma_points
    );

    /// @brief Kernel to resample the particles from predicted mean and covarience
    __global__ void resample_particles_mean_cov(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz,
        const float* predicted_mean,
        const float* predicted_cov,
        size_t num_particles,
        size_t n,
        unsigned int seed
    );

    /// @brief Function to call to resample particles using the kernel using predicted mean and covarience 
    void resampleParticlesMeanCov(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz,
        const float* d_predicted_mean,
        const float* d_predicted_cov,
        size_t num_particles,
        unsigned int seed 
    );

    /// @brief Function to run the prediction step
    void predict(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz, float* d_weights,
        float* d_mean, float* d_covariance, size_t num_particles,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz
    );

    /// @brief Kernel to resample particle indices
    __global__ void resample_particles(
        const float* d_cumulative_sum,  
        const float* d_random_numbers,  
        int* d_resampled_indices,       
        size_t num_particles
    );

    /// @brief Function to resample particles
    void resampleParticles(
        float* d_weights,  
        float* d_x,              
        float* d_y,              
        float* d_vx,             
        float* d_vy,             
        float* d_w,              
        float* d_h,              
        float* d_qw,             
        float* d_qx,             
        float* d_qy,             
        float* d_qz,             
        size_t num_particles     
    );

    /// @brief Kernel to update weights
    __global__ void compute_weights(
        const float* d_particles_x,
        const float* d_particles_y,
        const float obs_x,
        const float obs_y,
        float* d_weights,
        const size_t num_particles
    );

    /// @brief Kernel to update the particles 
    __global__ void update_particles(
        float* d_particles_x,
        float* d_particles_y,
        float* d_particles_vx,
        float* d_particles_vy,
        float* d_particles_w,
        float* d_particles_h,
        const float obs_x,
        const float obs_y,
        const float obs_vx,
        const float obs_vy,
        const float obs_w,
        const float obs_h,
        const float noise_std_x,  
        const float noise_std_y,  
        const float noise_std_vx, 
        const float noise_std_vy, 
        const float noise_std_w,  
        const float noise_std_h,  
        const size_t num_particles,
        unsigned int seed
    );

    /// @brief Kernel to update the quaternions
    __global__ void update_quaternions(
        float* d_particles_qw,
        float* d_particles_qx,
        float* d_particles_qy,
        float* d_particles_qz,
        const float obs_qw,
        const float obs_qx,
        const float obs_qy,
        const float obs_qz,
        const size_t num_particles
    );

    /// @brief Function to update the particles
    void updateParticles(
        float* d_particles_x,
        float* d_particles_y,
        float* d_particles_vx,
        float* d_particles_vy,
        float* d_particles_w,
        float* d_particles_h,
        float* d_particles_qw,
        float* d_particles_qx,
        float* d_particles_qy,
        float* d_particles_qz,
        float* d_weights,
        const float obs_x,
        const float obs_y,
        const float obs_vx,
        const float obs_vy,
        const float obs_w,
        const float obs_h,
        const float obs_qw,
        const float obs_qx,
        const float obs_qy,
        const float obs_qz,
        const float noise_std_x,
        const float noise_std_y,
        const float noise_std_vx,
        const float noise_std_vy,
        const float noise_std_w,
        const float noise_std_h,
        const size_t num_particles,
        const unsigned int seed
    );
}