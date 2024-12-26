#include "filter/kernels.cuh"

namespace particle_filter_kernels
{
    // Kernel to initialize particles with a normal distribution
    __global__ void initialize_particles(
        float* x, float* y, float* vx, float* vy, float* w, float* h, float* qw, float* qx, float* qy, float* qz,
        float* weights,
        float initial_x, float initial_y, float initial_vx, float initial_vy, float initial_w, float initial_h, float initial_qw, float initial_qx, float initial_qy, float initial_qz,
        size_t num_particles,
        float stddev_x, float stddev_y, float stddev_vx, float stddev_vy, float stddev_w, float stddev_h, float stddev_qw, float stddev_qx, float stddev_qy, float stddev_qz,
        unsigned int seed
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Set up CURAND state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate normal random offsets for x and y
        float offset_x = curand_normal(&state) * stddev_x;
        float offset_y = curand_normal(&state) * stddev_y;
        float offset_vx = curand_normal(&state) * stddev_vx;
        float offset_vy = curand_normal(&state) * stddev_vy;
        float offset_w = curand_normal(&state) * stddev_w;
        float offset_h = curand_normal(&state) * stddev_h;
        float offset_qw = curand_normal(&state) * stddev_qw;
        float offset_qx = curand_normal(&state) * stddev_qx;
        float offset_qy = curand_normal(&state) * stddev_qy;
        float offset_qz = curand_normal(&state) * stddev_qz;

        // Initialize x and y with normal-distributed offsets around initial values
        x[idx] = initial_x + offset_x;
        y[idx] = initial_y + offset_y;
        vx[idx] = initial_vx + offset_vx;
        vy[idx] = initial_vy + offset_vy;
        w[idx] = initial_w + offset_w;
        h[idx] = initial_h + offset_h;

        qw[idx] = initial_qw + offset_qw;
        qx[idx] = initial_qx + offset_qx;
        qy[idx] = initial_qy + offset_qy;
        qz[idx] = initial_qz + offset_qz;

        // Normalize the quaternion
        float norm = sqrtf(qw[idx] * qw[idx] + qx[idx] * qx[idx] + qy[idx] * qy[idx] + qz[idx] * qz[idx]);
        if (norm > 0.0f) 
        {   // Normalize if norm is valid
            qw[idx] /= norm;
            qx[idx] /= norm;
            qy[idx] /= norm;
            qz[idx] /= norm;
        } 
        else 
        {   // Fallback to default quaternion
            qw[idx] = 1.0f;
            qx[idx] = 0.0f;
            qy[idx] = 0.0f;
            qz[idx] = 0.0f;
        }

        // Initialize weights
        weights[idx] = 1.0f / num_particles;
    }

    // Function to launch the kernel
    void initParticles(
        float* d_x, float* d_y, float* d_vx, float* d_vy, float* d_w, float* d_h, float* d_qw, float* d_qx, float* d_qy, float* d_qz,
        float* d_weights,
        float initial_x, float initial_y, float initial_vx, float initial_vy, float initial_w, float initial_h, float initial_qw, float initial_qx, float initial_qy, float initial_qz,
        size_t num_particles,
        float stddev_x, float stddev_y, float stddev_vx, float stddev_vy, float stddev_w, float stddev_h, float stddev_qw, float stddev_qx, float stddev_qy, float stddev_qz,
        unsigned int seed
    )
    {
        // Set the kernel launch parameters
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        particle_filter_kernels::initialize_particles<<<blocks_per_grid, threads_per_block>>>(
            d_x, d_y, d_vx, d_vy, d_w, d_h, d_qw, d_qx, d_qy, d_qz,
            d_weights,
            initial_x, initial_y, initial_vx, initial_vy, initial_w, initial_h, initial_qw, initial_qx, initial_qy, initial_qz,
            num_particles, stddev_x, stddev_y, stddev_vx, stddev_vy, stddev_w, stddev_h, stddev_qw, stddev_qx, stddev_qy, stddev_qz, seed
        );

        // Synchronize to ensure the kernel has completed
        cudaDeviceSynchronize();
    }

    __global__ void compute_mean(
        const float* x, const float* y, const float* vx, const float* vy,
        const float* w, const float* h, const float* qw, const float* qx,
        const float* qy, const float* qz, const float* weights,
        float* mean, size_t num_particles
    )
    {
        __shared__ float shared_mem[10 * 256]; // Shared memory for reduction, 10 states × threads per block

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Initialize shared memory for each state
        for (int state_idx = 0; state_idx < 10; ++state_idx) 
        {
            shared_mem[state_idx * blockDim.x + tid] = 0.0f;
        }
        __syncthreads();

        // Accumulate weighted values for each particle
        if (idx < num_particles) 
        {
            float weight = weights[idx];
            shared_mem[tid] += weight * x[idx];
            shared_mem[blockDim.x + tid] += weight * y[idx];
            shared_mem[2 * blockDim.x + tid] += weight * vx[idx];
            shared_mem[3 * blockDim.x + tid] += weight * vy[idx];
            shared_mem[4 * blockDim.x + tid] += weight * w[idx];
            shared_mem[5 * blockDim.x + tid] += weight * h[idx];
            shared_mem[6 * blockDim.x + tid] += weight * qw[idx];
            shared_mem[7 * blockDim.x + tid] += weight * qx[idx];
            shared_mem[8 * blockDim.x + tid] += weight * qy[idx];
            shared_mem[9 * blockDim.x + tid] += weight * qz[idx];
        }
        __syncthreads();

        // Perform parallel reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
        {
            if (tid < stride) 
            {
                for (int state_idx = 0; state_idx < 10; ++state_idx) 
                {
                    shared_mem[state_idx * blockDim.x + tid] += shared_mem[state_idx * blockDim.x + tid + stride];
                }
            }
            __syncthreads();
        }

        // Write block-level results to global memory
        if (tid == 0) 
        {
            for (int state_idx = 0; state_idx < 10; ++state_idx) 
            {
                atomicAdd(&mean[state_idx], shared_mem[state_idx * blockDim.x]);
            }
        }
    }

    void computeMean(
        const float* d_x, const float* d_y, const float* d_vx, const float* d_vy,
        const float* d_w, const float* d_h, const float* d_qw, const float* d_qx,
        const float* d_qy, const float* d_qz, const float* d_weights,
        float* d_mean, size_t num_particles
    )
    {
        // Initialize the mean vector on the device to zero
        cudaError_t err = cudaMemset(d_mean, 0, 10 * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "CUDA error during cudaMemset: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Configure block and grid dimensions
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        particle_filter_kernels::compute_mean<<<blocks_per_grid, threads_per_block>>>(
            d_x, d_y, d_vx, d_vy, d_w, d_h, d_qw, d_qx, d_qy, d_qz, d_weights,
            d_mean, num_particles
        );

        // Synchronize to ensure kernel execution finishes
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
        {
            std::cerr << "CUDA error during cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
        }
    }

    __global__ void compute_covariance(
        const float* x, const float* y, const float* vx, const float* vy,
        const float* w, const float* h, const float* qw, const float* qx,
        const float* qy, const float* qz, const float* weights,
        const float* mean, float* covariance, size_t num_particles
    )
    {
        __shared__ float shared_mem[256]; // Shared memory for partial reductions

        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // State variables
        float states[10];
        if (idx < num_particles) 
        {
            float weight = weights[idx];
            states[0] = (x[idx] - mean[0]) * weight;
            states[1] = (y[idx] - mean[1]) * weight;
            states[2] = (vx[idx] - mean[2]) * weight;
            states[3] = (vy[idx] - mean[3]) * weight;
            states[4] = (w[idx] - mean[4]) * weight;
            states[5] = (h[idx] - mean[5]) * weight;
            states[6] = (qw[idx] - mean[6]) * weight;
            states[7] = (qx[idx] - mean[7]) * weight;
            states[8] = (qy[idx] - mean[8]) * weight;
            states[9] = (qz[idx] - mean[9]) * weight;
        } 
        else 
        {
            for (int i = 0; i < 10; ++i) 
            {
                states[i] = 0.0f;
            }
        }

        // Compute covariance elements sequentially
        for (int i = 0; i < 10; ++i) 
        {
            for (int j = 0; j <= i; ++j) // Compute lower triangular matrix
            { 
                // Load partial product into shared memory
                shared_mem[tid] = (idx < num_particles) ? states[i] * states[j] : 0.0f;
                __syncthreads();

                // Perform reduction in shared memory
                for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
                {
                    if (tid < stride) 
                    {
                        shared_mem[tid] += shared_mem[tid + stride];
                    }
                    __syncthreads();
                }

                // Write the result to global memory
                if (tid == 0) 
                {
                    atomicAdd(&covariance[i * 10 + j], shared_mem[0]);
                }
                __syncthreads();
            }
        }
    }

    __global__ void symmetrize_covariance(float* covariance, size_t n) 
    {
        int i = blockIdx.x;
        int j = threadIdx.x;

        if (i < n && j < n) 
        {
            if (i == j) 
            {
                // Add diagonal regularization
                covariance[i * n + j] += REGULARIZATION;
            } 
            else if (j > i) 
            {
                // Symmetrize the matrix
                float avg = (covariance[i * n + j] + covariance[j * n + i]) / 2.0f;
                covariance[i * n + j] = avg;
                covariance[j * n + i] = avg;
            }
        }

    }

    void computeCovariance(
        const float* d_x, const float* d_y, const float* d_vx, const float* d_vy,
        const float* d_w, const float* d_h, const float* d_qw, const float* d_qx,
        const float* d_qy, const float* d_qz, const float* d_weights,
        const float* d_mean, float* d_covariance, size_t num_particles
    )
    {
        // Initialize the covariance matrix on the GPU to zero
        cudaError_t err = cudaMemset(d_covariance, 0, 100 * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "CUDA error during cudaMemset: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Configure block and grid dimensions
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        particle_filter_kernels::compute_covariance<<<blocks_per_grid, threads_per_block>>>(
            d_x, d_y, d_vx, d_vy, d_w, d_h, d_qw, d_qx, d_qy, d_qz, d_weights,
            d_mean, d_covariance, num_particles
        );

        // Synchronize to ensure kernel execution finishes
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
        {
            std::cerr << "CUDA error during cudaDeviceSynchronize: " << cudaGetErrorString(err) << std::endl;
        }

        // Symmetrize the covariance matrix
        particle_filter_kernels::symmetrize_covariance<<<10, 10>>>(d_covariance, 10);

        // Synchronize again to ensure symmetrization finishes
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) 
        {
            std::cerr << "CUDA error during symmetrization: " << cudaGetErrorString(err) << std::endl;
        }
    }

    __global__ void compute_weights_sigma(
        size_t n, float lambda, float alpha, float beta,
        float* weights_mean, float* weights_cov
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx > 2 * n) return; // Only valid for [0, 2n]

        float scaling_factor = n + lambda;

        if (idx == 0) 
        {
            // Compute weights for the central sigma point
            weights_mean[idx] = lambda / scaling_factor;
            weights_cov[idx] = weights_mean[idx] + (1.0f - alpha * alpha + beta);
        } 
        else 
        {
            // Compute weights for the remaining sigma points
            weights_mean[idx] = 1.0f / (2.0f * scaling_factor);
            weights_cov[idx] = weights_mean[idx];
        }
    }

    void computeWeightsMeanCovSigma(
        size_t n, float alpha, float beta, float kappa,
        float* d_weights_mean, float* d_weights_cov
    )
    {
        // Compute lambda
        float lambda = alpha * alpha * (n + kappa) - n;

        // Launch kernel to compute weights
        int threads_per_block = 256;
        int blocks_per_grid = (2 * n + threads_per_block) / threads_per_block;

        particle_filter_kernels::compute_weights_sigma<<<blocks_per_grid, threads_per_block>>>(
            n, lambda, alpha, beta, d_weights_mean, d_weights_cov
        );

        cudaDeviceSynchronize(); // Ensure kernel execution is complete
    }

    void computeCholeskyDecomposition(
        const float* d_covariance,
        float* d_cholesky,
        size_t n
    ) 
    {
        cusolverDnHandle_t cusolverH = nullptr;
        cudaStream_t stream = nullptr;

        // Create cuSOLVER handle and stream
        cusolverDnCreate(&cusolverH);
        cudaStreamCreate(&stream);
        cusolverDnSetStream(cusolverH, stream);

        // Copy input covariance matrix to output buffer (to preserve original)
        cudaMemcpy(d_cholesky, d_covariance, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

        // Workspace and info buffer for Cholesky decomposition
        int workspace_size = 0;
        int* d_info = nullptr;
        float* d_workspace = nullptr;

        cudaMalloc(&d_info, sizeof(int));
        cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, n, d_cholesky, n, &workspace_size);
        cudaMalloc(&d_workspace, workspace_size * sizeof(float));

        // Perform Cholesky decomposition
        cusolverStatus_t status = cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, n, d_cholesky, n, d_workspace, workspace_size, d_info);

        if (status != CUSOLVER_STATUS_SUCCESS) 
        {
            std::cerr << "ERROR: Cholesky decomposition failed. Status = " << status << std::endl;
        }

        int h_info = 0;
        cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
        if (h_info > 0) {
            std::cerr << "ERROR: Cholesky decomposition failed. Matrix is not positive definite. Info = " << h_info << std::endl;
        } else if (h_info < 0) {
            std::cerr << "ERROR: Invalid argument for Cholesky decomposition. Info = " << h_info << std::endl;
        }

        // Cleanup
        cudaFree(d_info);
        cudaFree(d_workspace);
        cusolverDnDestroy(cusolverH);
        cudaStreamDestroy(stream);
    }


    __global__ void compute_sigma_points(
        const float* d_mean,
        const float* d_cholesky,
        float* d_sigma_points,
        size_t n
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Ensure we don't exceed the number of sigma points
        if (idx >= 2 * n + 1) return;

        if (idx == 0) 
        {
            // Central sigma point: sigma_points[0] = mean
            for (int i = 0; i < n; ++i) 
            {
                d_sigma_points[i] = d_mean[i];  // Correctly copy mean to first sigma point
            }
        } 
        else if (idx <= n) 
        {
            // Positive deviation: sigma_points[i+1] = mean + sqrt_matrix[row]
            int row = idx - 1;
            for (int col = 0; col < n; ++col) 
            {
                d_sigma_points[idx * n + col] = d_mean[col] + d_cholesky[row * n + col];
            }
        } 
        else 
        {
            // Negative deviation: sigma_points[n+i+1] = mean - sqrt_matrix[row]
            int row = idx - n - 1;
            for (int col = 0; col < n; ++col) 
            {
                d_sigma_points[idx * n + col] = d_mean[col] - d_cholesky[row * n + col];
            }
        }

        // Normalize quaternion states (6 to 9) if they are within the state vector range
        if (n > 6 && idx < 2 * n + 1) 
        {
            float norm = 0.0f;
            for (int i = 6; i < 10 && i < n; ++i) 
            {
                norm += d_sigma_points[idx * n + i] * d_sigma_points[idx * n + i];
            }
            norm = sqrtf(norm);
            if (norm > 1e-6f) 
            {
                for (int i = 6; i < 10 && i < n; ++i) 
                {
                    d_sigma_points[idx * n + i] /= norm;
                }
            }
        }
    }

    void computeSigmaPoints(
        const float* d_mean,         
        const float* d_cholesky,    
        float* d_sigma_points,       
        size_t n                     
    )
    {
        // Calculate number of sigma points
        size_t num_sigma_points = 2 * n + 1;

        // Define CUDA kernel launch configuration
        int threads_per_block = 256;
        int blocks_per_grid = (num_sigma_points + threads_per_block - 1) / threads_per_block;

        // Launch the compute_sigma_points kernel
        particle_filter_kernels::compute_sigma_points<<<blocks_per_grid, threads_per_block>>>(
            d_mean, d_cholesky, d_sigma_points, n
        );

        // Synchronize to ensure kernel execution completes
        cudaDeviceSynchronize();
    }

    thrust::tuple<thrust::host_vector<float>, thrust::host_vector<float>, thrust::host_vector<float>> unscentedTransform(
        const float* d_mean,
        const float* d_covariance,                                    
        float alpha,                                     
        float beta,                                      
        float kappa                                      
    )
    {
        const int n = 10;
        // Step 1: Compute weights
        thrust::device_vector<float> weights_mean(2 * n + 1);
        thrust::device_vector<float> weights_covariance(2 * n + 1);
        computeWeightsMeanCovSigma(
            n, alpha, beta, kappa,
            thrust::raw_pointer_cast(weights_mean.data()),
            thrust::raw_pointer_cast(weights_covariance.data())
        );

        // Step 2: Perform Cholesky decomposition
        thrust::device_vector<float> cholesky(n * n);
        computeCholeskyDecomposition(
            d_covariance,
            thrust::raw_pointer_cast(cholesky.data()),
            n
        );

        // Step 3: Generate sigma points
        thrust::device_vector<float> sigma_points((2 * n + 1) * n);
        computeSigmaPoints(
            d_mean,
            thrust::raw_pointer_cast(cholesky.data()),
            thrust::raw_pointer_cast(sigma_points.data()),
            n
        );

        // Step 4 : Make host vector for this part
        thrust::host_vector<float> h_sigma = sigma_points;
        thrust::host_vector<float> h_weights_mean = weights_mean;
        thrust::host_vector<float> h_weights_cov = weights_covariance;

        // Return the tuple containing sigma points, weights for mean, and weights for covariance
        return thrust::make_tuple(h_sigma, h_weights_mean, h_weights_cov);
    }

    __global__ void propagate_sigma_points(
        float* sigma_points,
        size_t num_sigma_points,
        size_t n,
        float dt,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz,
        unsigned int seed
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_sigma_points) return; // Out-of-bounds check

        // Initialize CURAND state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Pointer to the specific sigma point in the flattened array
        float* sigma = sigma_points + idx * n;

        // Extract current state
        float x = sigma[0];
        float y = sigma[1];
        float vx = sigma[2];
        float vy = sigma[3];
        float w = sigma[4];
        float h = sigma[5];
        float qw = sigma[6];
        float qx = sigma[7];
        float qy = sigma[8];
        float qz = sigma[9];

        // Update position with velocity and noise
        x += vx * dt + noise_x * curand_normal(&state);
        y += vy * dt + noise_y * curand_normal(&state);

        // Update size with noise
        w += noise_w * curand_normal(&state);
        h += noise_h * curand_normal(&state);

        // Update quaternion with noise
        qw += noise_qw * curand_normal(&state);
        qx += noise_qx * curand_normal(&state);
        qy += noise_qy * curand_normal(&state);
        qz += noise_qz * curand_normal(&state);

        // Normalize quaternion
        float norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
        if (norm > 0.0f) 
        {
            qw /= norm;
            qx /= norm;
            qy /= norm;
            qz /= norm;
        } 
        else 
        {
            // Fallback to default quaternion
            qw = 1.0f;
            qx = 0.0f;
            qy = 0.0f;
            qz = 0.0f;
        }

        // Write updated state back to sigma point
        sigma[0] = x;
        sigma[1] = y;
        sigma[2] = vx;
        sigma[3] = vy;
        sigma[4] = w;
        sigma[5] = h;
        sigma[6] = qw;
        sigma[7] = qx;
        sigma[8] = qy;
        sigma[9] = qz;
    }

    void propagateSigmaPoints(
        float* d_sigma_points,
        float dt,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz,
        unsigned int seed
    )
    {

        const int n = 10;
        const int num_sigma_points = (2 * 10 + 1) * 10;
        // Configure kernel launch parameters
        int threads_per_block = 256;
        int blocks_per_grid = (num_sigma_points + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        particle_filter_kernels::propagate_sigma_points<<<blocks_per_grid, threads_per_block>>>(
            d_sigma_points,
            num_sigma_points,
            n,
            dt,
            noise_x, noise_y,
            noise_w, noise_h,
            noise_qw, noise_qx, noise_qy, noise_qz,
            seed
        );

        // Synchronize to ensure the kernel finishes
        cudaDeviceSynchronize();
    }

    __global__ void compute_mean_propogated(
        float* sigma_points,
        float* weights_mean,
        float* predicted_mean,
        size_t num_sigma_points,
        size_t n
    )
    {
        int state_idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Exit early if the thread index is out of bounds
        if (state_idx >= n) return;

        float weighted_sum = 0.0f;

        // Compute weighted sum for this state across all sigma points
        for (size_t sigma_idx = 0; sigma_idx < num_sigma_points; ++sigma_idx) 
        {
            // Properly index into the sigma points array
            float sigma_value = sigma_points[sigma_idx * n + state_idx];
            float weight = weights_mean[sigma_idx];

            // Accumulate weighted value
            weighted_sum += weight * sigma_value;
        }

        // Store the result in the predicted mean
        predicted_mean[state_idx] = weighted_sum;

        // Normalize quaternion if this thread corresponds to a quaternion component
        if (state_idx == 6 || state_idx == 7 || state_idx == 8 || state_idx == 9) // Quaternion indices
        {
            __shared__ float quaternion_norm; // Shared memory for norm calculation

            // Compute squared norm for quaternion components
            if (threadIdx.x == 0) quaternion_norm = 0.0f;
            __syncthreads();

            atomicAdd(&quaternion_norm, predicted_mean[state_idx] * predicted_mean[state_idx]);
            __syncthreads();

            if (threadIdx.x == 0) quaternion_norm = sqrtf(quaternion_norm);
            __syncthreads();

            // Normalize quaternion components
            predicted_mean[state_idx] /= quaternion_norm;
        }
    }

    __global__ void subtract_mean(
        const float* sigma_points,    // Input sigma points (num_sigma_points x n)
        const float* predicted_mean, // Predicted mean vector (n)
        float* delta_sigma_points,   // Output: sigma_points - mean (num_sigma_points x n)
        size_t num_sigma_points,     // Number of sigma points
        size_t n                     // Number of states
    ) {
        // Compute the thread's index
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        // Each thread handles one element in the sigma points array
        if (idx < num_sigma_points * n) 
        {
            int sigma_idx = idx / n;  // Sigma point index
            int state_idx = idx % n; // State index within the sigma point

            // Access using 2D representation for clarity
            delta_sigma_points[sigma_idx * n + state_idx] =
                sigma_points[sigma_idx * n + state_idx] - predicted_mean[state_idx];
        }
    }

    __global__ void scale_delta_sigma_points(
        float* d_delta_sigma_points,
        const float* d_weights_cov,
        size_t num_sigma_points,
        size_t n
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_sigma_points * n) return;

        int col_idx = idx / n; // Column index corresponds to the sigma point

        float weight_sqrt = sqrtf(d_weights_cov[col_idx]);
        d_delta_sigma_points[idx] *= weight_sqrt;
    }

    void computeCovarianceWithCuBLAS(
        cublasHandle_t cublas_handle,
        const float* d_delta_sigma_points, // Input: Scaled deviations
        float* d_predicted_covariance,    // Output: Covariance matrix
        size_t num_sigma_points,
        size_t n                          // Number of states
    )
    {
        const float alpha = 0.005f;
        const float beta = 0.0f;

        // Transpose the Delta Sigma Points: d_delta_sigma_points^T
        // Compute: Cov = (Delta_Sigma_Points^T) x Delta_Sigma_Points
        // cuBLAS is column-major, so we need to "reverse" the dimensions for GEMM:
        // d_predicted_covariance = d_delta_sigma_points^T x d_delta_sigma_points
        cublasStatus_t status = cublasSgemm(
            cublas_handle,
            CUBLAS_OP_T,            // Transpose the first matrix
            CUBLAS_OP_N,            // Do not transpose the second matrix
            n,                      // Rows of Covariance (and Delta Sigma Transposed)
            n,                      // Columns of Covariance
            num_sigma_points,       // Inner dimension: Columns of Delta Sigma Points
            &alpha,                 // Scaling factor for the multiplication
            d_delta_sigma_points,   // Pointer to Delta Sigma Points (A matrix)
            num_sigma_points,       // Leading dimension of A (number of rows)
            d_delta_sigma_points,   // Pointer to Delta Sigma Points (B matrix)
            num_sigma_points,       // Leading dimension of B (number of rows)
            &beta,                  // Scaling factor for Covariance matrix
            d_predicted_covariance, // Pointer to Covariance matrix (C matrix)
            n                       // Leading dimension of Covariance matrix
        );

        // Check for errors
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            printf("ERROR: cuBLAS SGEMM failed! Status = %d\n", status);
        }
    }

    void computeMeanAndCovariance(
        float* d_sigma_points,
        float* d_weights_mean,
        float* d_weights_cov,
        float* d_predicted_mean,
        float* d_predicted_covariance,
        float* d_delta_sigma_points
    )
    {
        const int n = 10;
        const int num_sigma_points = 2 * n + 1;

        // Mean computation
        int threads_per_block = 256;
        int blocks_per_grid_mean = (n + threads_per_block - 1) / threads_per_block;

        particle_filter_kernels::compute_mean_propogated<<<blocks_per_grid_mean, threads_per_block>>>(
            d_sigma_points,
            d_weights_mean,
            d_predicted_mean,
            num_sigma_points,
            n
        );

        cudaDeviceSynchronize();

        int blocks_per_grid = (num_sigma_points * n + threads_per_block - 1) / threads_per_block;
        particle_filter_kernels::subtract_mean<<<blocks_per_grid, threads_per_block>>>(d_sigma_points, d_predicted_mean, d_delta_sigma_points, num_sigma_points, n);

        cudaDeviceSynchronize();

        size_t total_elements = num_sigma_points * n;
        threads_per_block = 256;
        blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        particle_filter_kernels::scale_delta_sigma_points<<<blocks_per_grid, threads_per_block>>>(d_delta_sigma_points, d_weights_cov, num_sigma_points, n);

        // Synchronize after kernel calls
        cudaDeviceSynchronize();

        cublasHandle_t cublas_handle;
        cublasStatus_t status = cublasCreate(&cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            std::cerr << "ERROR: cuBLAS handle creation failed! Status = " << status << std::endl;
        }
        particle_filter_kernels::computeCovarianceWithCuBLAS(cublas_handle, d_delta_sigma_points, d_predicted_covariance, num_sigma_points, n);

        status = cublasDestroy(cublas_handle);

        if (status != CUBLAS_STATUS_SUCCESS) 
        {
            std::cerr << "ERROR: cuBLAS handle destruction failed!" << std::endl;
        }
    }

    __global__ void resample_particles_mean_cov(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz,
        const float* predicted_mean,
        const float* predicted_cov,
        size_t num_particles,
        size_t n,
        unsigned int seed
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        // Initialize CURAND state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate Gaussian noise (standard normal)
        float* noise = new float[n];
        for (int i = 0; i < n; ++i) 
        {
            noise[i] = curand_normal(&state);
        }

        // Transform noise to match predicted mean and covariance
        float particle_state[10]; // Temporary state array
        for (int i = 0; i < n; ++i) 
        {
            particle_state[i] = predicted_mean[i];
            for (int j = 0; j <= i; ++j) // Use lower triangular part of covariance
            { 
                particle_state[i] += predicted_cov[i * n + j] * noise[j];
            }
        }

        delete[] noise;

        // Normalize quaternion (elements 6 to 9)
        float qw = particle_state[6], qx = particle_state[7], qy = particle_state[8], qz = particle_state[9];
        float norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz);
        if (norm > 0.0f) 
        {
            particle_state[6] /= norm;
            particle_state[7] /= norm;
            particle_state[8] /= norm;
            particle_state[9] /= norm;
        }
        else
        {
            particle_state[6] = 1.0f;
            particle_state[7] = 0.0f;
            particle_state[8] = 0.0f;
            particle_state[9] = 0.0f;
        }

        // Write the updated state back to the individual arrays
        d_x[idx] = particle_state[0];
        d_y[idx] = particle_state[1];
        d_vx[idx] = particle_state[2];
        d_vy[idx] = particle_state[3];
        d_w[idx] = particle_state[4];
        d_h[idx] = particle_state[5];
        d_qw[idx] = particle_state[6];
        d_qx[idx] = particle_state[7];
        d_qy[idx] = particle_state[8];
        d_qz[idx] = particle_state[9];
    }

    void resampleParticlesMeanCov(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz,
        const float* d_predicted_mean,
        const float* d_predicted_cov,
        size_t num_particles,
        unsigned int seed
    )
    {
        // Dimensionality of the state vector (hardcoded for now, can be parameterized)
        const size_t n = 10;

        // Configure kernel launch parameters
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

        // Launch the kernel
        particle_filter_kernels::resample_particles_mean_cov<<<blocks_per_grid, threads_per_block>>>(
            d_x, d_y, d_vx, d_vy,
            d_w, d_h, d_qw, d_qx,
            d_qy, d_qz,
            d_predicted_mean,
            d_predicted_cov,
            num_particles,
            n,
            seed
        );

        // Synchronize to ensure the kernel finishes
        cudaDeviceSynchronize();
    }

    void predict(
        float* d_x, float* d_y, float* d_vx, float* d_vy,
        float* d_w, float* d_h, float* d_qw, float* d_qx,
        float* d_qy, float* d_qz, float* d_weights,
        float* d_mean, float* d_covariance, size_t num_particles,
        float noise_x, float noise_y,
        float noise_w, float noise_h,
        float noise_qw, float noise_qx, float noise_qy, float noise_qz
    )
    {
        // Step 1: Compute Mean
        particle_filter_kernels::computeMean(d_x, d_y, d_vx, d_vy,
            d_w, d_h, d_qw, d_qx,
            d_qy, d_qz, d_weights,
            d_mean, num_particles
        );

        // Step 2: Compute Covariance
        particle_filter_kernels::computeCovariance(d_x, d_y, d_vx, d_vy,
            d_w, d_h, d_qw, d_qx,
            d_qy, d_qz, d_weights,
            d_mean, d_covariance, num_particles
        );

        // Step 3: Compute Unscented Transform
        thrust::tuple<thrust::host_vector<float>, thrust::host_vector<float>, thrust::host_vector<float>> UT = particle_filter_kernels::unscentedTransform(d_mean, d_covariance, 0.5, 2, 3);

        // Extract the components from the tuple
        thrust::host_vector<float>& sigma_points = thrust::get<0>(UT);
        thrust::host_vector<float>& weights_mean = thrust::get<1>(UT);
        thrust::host_vector<float>& weights_covariance = thrust::get<2>(UT);

        size_t num_states = 10; 
        size_t num_sigma_points = 2 * num_states + 1;

        thrust::device_vector<float> d_sigma_points = sigma_points;
        
        // Step 4: Propogate Sigma Points
        particle_filter_kernels::propagateSigmaPoints(thrust::raw_pointer_cast(d_sigma_points.data()), 1.0, 
            noise_x,noise_y,
            noise_w, noise_h,
            noise_qw, noise_qx, noise_qy, noise_qz, 
            static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())
        );
        
        thrust::device_vector<float> d_weights_mean = weights_mean;
        thrust::device_vector<float> d_weights_cov = weights_covariance;
        thrust::device_vector<float> d_delta_sigma_points(num_sigma_points * 10);

        // Step 5: Compute Mean and Covariance of propogated sigma points
        particle_filter_kernels::computeMeanAndCovariance(thrust::raw_pointer_cast(d_sigma_points.data()), thrust::raw_pointer_cast(d_weights_mean.data()), thrust::raw_pointer_cast(d_weights_cov.data()), d_mean, d_covariance, thrust::raw_pointer_cast(d_delta_sigma_points.data()));

        // Step 6: Resample particles
        particle_filter_kernels::resampleParticlesMeanCov(d_x, d_y, d_vx, d_vy,
            d_w, d_h, d_qw, d_qx,
            d_qy, d_qz, 
            d_mean, d_covariance, num_particles,
            static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count())
        );
    }

    __global__ void resample_particles(
        const float* d_cumulative_sum,  
        const float* d_random_numbers,  
        int* d_resampled_indices,       
        size_t num_particles
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Find the index of the cumulative sum that corresponds to the random number
        float random_value = d_random_numbers[idx];
        int resampled_index = 0;

        for (int i = 0; i < num_particles; ++i) 
        {
            if (random_value <= d_cumulative_sum[i]) 
            {
                resampled_index = i;
                break;
            }
        }

        d_resampled_indices[idx] = resampled_index;
    }

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
    ) 
    {
        // Allocate memory for cumulative sum and random numbers
        float* d_cumulative_sum;
        cudaMalloc(&d_cumulative_sum, num_particles * sizeof(float));

        float* d_random_numbers;
        cudaMalloc(&d_random_numbers, num_particles * sizeof(float));

        // Compute cumulative sum of weights
        thrust::device_ptr<float> weights_ptr(d_weights);
        thrust::device_ptr<float> cumulative_sum_ptr(d_cumulative_sum);
        thrust::inclusive_scan(weights_ptr, weights_ptr + num_particles, cumulative_sum_ptr);

        // Generate random numbers
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
        curandGenerateUniform(gen, d_random_numbers, num_particles);

        // Allocate memory for resampled indices
        int* d_resampled_indices;
        cudaMalloc(&d_resampled_indices, num_particles * sizeof(int));

        // Launch kernel to resample indices
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;
        particle_filter_kernels::resample_particles<<<blocks_per_grid, threads_per_block>>>(
            d_cumulative_sum,
            d_random_numbers,
            d_resampled_indices,
            num_particles
        );

        // Resample all particle states using the resampled indices
        thrust::device_ptr<int> resampled_indices_ptr(d_resampled_indices);

        // Temporary arrays for resampled states
        float* d_resampled_x; cudaMalloc(&d_resampled_x, num_particles * sizeof(float));
        float* d_resampled_y; cudaMalloc(&d_resampled_y, num_particles * sizeof(float));
        float* d_resampled_vx; cudaMalloc(&d_resampled_vx, num_particles * sizeof(float));
        float* d_resampled_vy; cudaMalloc(&d_resampled_vy, num_particles * sizeof(float));
        float* d_resampled_w; cudaMalloc(&d_resampled_w, num_particles * sizeof(float));
        float* d_resampled_h; cudaMalloc(&d_resampled_h, num_particles * sizeof(float));
        float* d_resampled_qw; cudaMalloc(&d_resampled_qw, num_particles * sizeof(float));
        float* d_resampled_qx; cudaMalloc(&d_resampled_qx, num_particles * sizeof(float));
        float* d_resampled_qy; cudaMalloc(&d_resampled_qy, num_particles * sizeof(float));
        float* d_resampled_qz; cudaMalloc(&d_resampled_qz, num_particles * sizeof(float));

        // Resample each state array
        thrust::device_ptr<float> x_ptr(d_x), resampled_x_ptr(d_resampled_x);
        thrust::device_ptr<float> y_ptr(d_y), resampled_y_ptr(d_resampled_y);
        thrust::device_ptr<float> vx_ptr(d_vx), resampled_vx_ptr(d_resampled_vx);
        thrust::device_ptr<float> vy_ptr(d_vy), resampled_vy_ptr(d_resampled_vy);
        thrust::device_ptr<float> w_ptr(d_w), resampled_w_ptr(d_resampled_w);
        thrust::device_ptr<float> h_ptr(d_h), resampled_h_ptr(d_resampled_h);
        thrust::device_ptr<float> qw_ptr(d_qw), resampled_qw_ptr(d_resampled_qw);
        thrust::device_ptr<float> qx_ptr(d_qx), resampled_qx_ptr(d_resampled_qx);
        thrust::device_ptr<float> qy_ptr(d_qy), resampled_qy_ptr(d_resampled_qy);
        thrust::device_ptr<float> qz_ptr(d_qz), resampled_qz_ptr(d_resampled_qz);

        for (size_t i = 0; i < num_particles; ++i) 
        {
            int idx = resampled_indices_ptr[i];
            resampled_x_ptr[i] = x_ptr[idx];
            resampled_y_ptr[i] = y_ptr[idx];
            resampled_vx_ptr[i] = vx_ptr[idx];
            resampled_vy_ptr[i] = vy_ptr[idx];
            resampled_w_ptr[i] = w_ptr[idx];
            resampled_h_ptr[i] = h_ptr[idx];
            resampled_qw_ptr[i] = qw_ptr[idx];
            resampled_qx_ptr[i] = qx_ptr[idx];
            resampled_qy_ptr[i] = qy_ptr[idx];
            resampled_qz_ptr[i] = qz_ptr[idx];
        }

        // Copy resampled particles back to the original arrays
        cudaMemcpy(d_x, d_resampled_x, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_y, d_resampled_y, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vx, d_resampled_vx, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vy, d_resampled_vy, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_w, d_resampled_w, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_h, d_resampled_h, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_qw, d_resampled_qw, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_qx, d_resampled_qx, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_qy, d_resampled_qy, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_qz, d_resampled_qz, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);

        // Normalize weights
        float sum_weights = thrust::reduce(weights_ptr, weights_ptr + num_particles, 0.0f, thrust::plus<float>());
        if (sum_weights > 0.0f) 
        {
            thrust::transform(weights_ptr, weights_ptr + num_particles, weights_ptr, [sum_weights] __device__(float w) 
            {
                return w / sum_weights;
            });
        }

        // Cleanup
        curandDestroyGenerator(gen);
        cudaFree(d_cumulative_sum);
        cudaFree(d_random_numbers);
        cudaFree(d_resampled_indices);
        cudaFree(d_resampled_x);
        cudaFree(d_resampled_y);
        cudaFree(d_resampled_vx);
        cudaFree(d_resampled_vy);
        cudaFree(d_resampled_w);
        cudaFree(d_resampled_h);
        cudaFree(d_resampled_qw);
        cudaFree(d_resampled_qx);
        cudaFree(d_resampled_qy);
        cudaFree(d_resampled_qz);
    }


    __global__ void compute_weights(
        const float* d_particles_x,
        const float* d_particles_y,
        const float obs_x,
        const float obs_y,
        float* d_weights,
        const size_t num_particles
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Compute Euclidean distance to the observation
        float dx = d_particles_x[idx] - obs_x;
        float dy = d_particles_y[idx] - obs_y;
        float distance = sqrtf(dx * dx + dy * dy);

        // Update weight using exponential proximity
        d_weights[idx] = expf(-distance / 10.0f) + 1e-10f;
    }

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
    )
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Add weighted blending with noise
        d_particles_x[idx] = 0.5f * d_particles_x[idx] + 0.5f * obs_x + curand_normal(&state) * noise_std_x;
        d_particles_y[idx] = 0.5f * d_particles_y[idx] + 0.5f * obs_y + curand_normal(&state) * noise_std_y;
        d_particles_vx[idx] = 0.5f * d_particles_vx[idx] + 0.5f * obs_vx + curand_normal(&state) * noise_std_vx;
        d_particles_vy[idx] = 0.5f * d_particles_vy[idx] + 0.5f * obs_vy + curand_normal(&state) * noise_std_vy;
        d_particles_w[idx] = 0.5f * d_particles_w[idx] + 0.5f * obs_w + curand_normal(&state) * noise_std_w;
        d_particles_h[idx] = 0.5f * d_particles_h[idx] + 0.5f * obs_h + curand_normal(&state) * noise_std_h;
    }

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
    ) 
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_particles) return;

        // Current particle quaternion
        float current_qw = d_particles_qw[idx];
        float current_qx = d_particles_qx[idx];
        float current_qy = d_particles_qy[idx];
        float current_qz = d_particles_qz[idx];

        // Compute dot product to determine if we need to invert the observation quaternion
        float dot = current_qw * obs_qw + current_qx * obs_qx + current_qy * obs_qy + current_qz * obs_qz;
        float interp_obs_qw = obs_qw;
        float interp_obs_qx = obs_qx;
        float interp_obs_qy = obs_qy;
        float interp_obs_qz = obs_qz;

        // If dot product is negative, negate the observation quaternion to take the shortest path
        if (dot < 0.0f) 
        {
            interp_obs_qw = -obs_qw;
            interp_obs_qx = -obs_qx;
            interp_obs_qy = -obs_qy;
            interp_obs_qz = -obs_qz;
            dot = -dot;
        }

        // Compute interpolation factor (e.g., SLERP with t = 0.5)
        float t = 0.5f;

        // Compute the interpolated quaternion
        float scale1, scale2;
        if (dot > 0.9995f) 
        {
            // Use linear interpolation for very close quaternions
            scale1 = 1.0f - t;
            scale2 = t;
        } else 
        {
            // Use SLERP
            float theta = acosf(dot);
            float sin_theta = sinf(theta);
            scale1 = sinf((1.0f - t) * theta) / sin_theta;
            scale2 = sinf(t * theta) / sin_theta;
        }

        float new_qw = scale1 * current_qw + scale2 * interp_obs_qw;
        float new_qx = scale1 * current_qx + scale2 * interp_obs_qx;
        float new_qy = scale1 * current_qy + scale2 * interp_obs_qy;
        float new_qz = scale1 * current_qz + scale2 * interp_obs_qz;

        // Normalize the interpolated quaternion
        float norm = sqrtf(new_qw * new_qw + new_qx * new_qx + new_qy * new_qy + new_qz * new_qz);

        if (norm > 0.0f) 
        {
            d_particles_qw[idx] = new_qw / norm;
            d_particles_qx[idx] = new_qx / norm;
            d_particles_qy[idx] = new_qy / norm;
            d_particles_qz[idx] = new_qz / norm;
        }
        else
        {
            d_particles_qw[idx] = 1.0f;
            d_particles_qx[idx] = 0.0f;
            d_particles_qy[idx] = 0.0f;
            d_particles_qz[idx] = 0.0f;
        }
    }

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
    ) 
    {
        // Define kernel launch parameters
        int threads_per_block = 256;
        int blocks_per_grid = (num_particles + threads_per_block - 1) / threads_per_block;

        // Step 1: Compute weights
        particle_filter_kernels::compute_weights<<<blocks_per_grid, threads_per_block>>>(
            d_particles_x,
            d_particles_y,
            obs_x,
            obs_y,
            d_weights,
            num_particles
        );
        cudaDeviceSynchronize();

        // Step 2: Update linear states (x, y, vx, vy, w, h) with blending and noise
        particle_filter_kernels::update_particles<<<blocks_per_grid, threads_per_block>>>(
            d_particles_x,
            d_particles_y,
            d_particles_vx,
            d_particles_vy,
            d_particles_w,
            d_particles_h,
            obs_x,
            obs_y,
            obs_vx,
            obs_vy,
            obs_w,
            obs_h,
            noise_std_x,
            noise_std_y,
            noise_std_vx,
            noise_std_vy,
            noise_std_w,
            noise_std_h,
            num_particles,
            seed
        );
        cudaDeviceSynchronize();

        // Step 3: Update quaternion states (qw, qx, qy, qz) with SLERP
        particle_filter_kernels::update_quaternions<<<blocks_per_grid, threads_per_block>>>(
            d_particles_qw,
            d_particles_qx,
            d_particles_qy,
            d_particles_qz,
            obs_qw,
            obs_qx,
            obs_qy,
            obs_qz,
            num_particles
        );
        cudaDeviceSynchronize();

        resampleParticles(
            d_weights,  
            d_particles_x,              
            d_particles_y,              
            d_particles_vx,             
            d_particles_vy,             
            d_particles_w,              
            d_particles_h,              
            d_particles_qw,             
            d_particles_qx,             
            d_particles_qy,             
            d_particles_qz,             
            num_particles     
        );
        cudaDeviceSynchronize();
    }
}