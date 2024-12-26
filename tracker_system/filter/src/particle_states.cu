#include "filter/particle_states.cuh"

namespace particle_filter
{
    // Constructor: Initialize device vectors with number of particles
    ParticleStates::ParticleStates(size_t num_particles) : num_particles_(num_particles)
    {
        this->resize_(num_particles);
    }

    size_t ParticleStates::size()
    {
        return this->num_particles_;
    }

    size_t ParticleStates::size_states()
    {
        return 10;
    }

    // Upload data from host vectors to device vectors
    void ParticleStates::upload()
    {
        this->d_x_ = this->h_x_;
        this->d_y_ = this->h_y_;
        this->d_vx_ = this->h_vx_;
        this->d_vy_ = this->h_vy_;
        this->d_w_ = this->h_w_;
        this->d_h_ = this->h_h_;
        this->d_qw_ = this->h_qw_;
        this->d_qx_ = this->h_qx_;
        this->d_qy_ = this->h_qy_;
        this->d_qz_ = this->h_qz_;
        this->d_mean_ = this->h_mean_;
        this->d_cov_ = this->h_cov_;
        this->d_weights_ = this->h_weights_;
    }

    // Download data from device vectors to host vectors
    void ParticleStates::download()
    {
        this->h_x_ = this->d_x_;
        this->h_y_ = this->d_y_;
        this->h_vx_ = this->d_vx_;
        this->h_vy_ = this->d_vy_;
        this->h_w_ = this->d_w_;
        this->h_h_ = this->d_h_;
        this->h_qw_ = this->d_qw_;
        this->h_qx_ = this->d_qx_;
        this->h_qy_ = this->d_qy_;
        this->h_qz_ = this->d_qz_;
        this->h_mean_ = this->d_mean_;
        this->h_cov_ = this->d_cov_;
        this->h_weights_ = this->d_weights_;
    }

    // Return raw pointer to device x vector
    float* ParticleStates::device_x()
    {
        return thrust::raw_pointer_cast(this->d_x_.data());
    }

    // Return raw pointer to device y vector
    float* ParticleStates::device_y()
    {
        return thrust::raw_pointer_cast(this->d_y_.data());
    }

    float* ParticleStates::device_vx()
    {
        return thrust::raw_pointer_cast(this->d_vx_.data());
    }

    float* ParticleStates::device_vy()
    {
        return thrust::raw_pointer_cast(this->d_vy_.data());
    }

    float* ParticleStates::device_w()
    {
        return thrust::raw_pointer_cast(this->d_w_.data());
    }

    float* ParticleStates::device_h()
    {
        return thrust::raw_pointer_cast(this->d_h_.data());
    }

    float* ParticleStates::device_qw()
    {
        return thrust::raw_pointer_cast(this->d_qw_.data());
    }

    float* ParticleStates::device_qx()
    {
        return thrust::raw_pointer_cast(this->d_qx_.data());
    }

    float* ParticleStates::device_qy()
    {
        return thrust::raw_pointer_cast(this->d_qy_.data());
    }

    float* ParticleStates::device_qz()
    {
        return thrust::raw_pointer_cast(this->d_qz_.data());
    }

    float* ParticleStates::device_mean()
    {
        return thrust::raw_pointer_cast(this->d_mean_.data());
    }

    float* ParticleStates::device_cov()
    {
        return thrust::raw_pointer_cast(this->d_cov_.data());
    }

    float* ParticleStates::device_weights()
    {
        return thrust::raw_pointer_cast(this->d_weights_.data());
    }

    float* ParticleStates::host_x()
    {
        return this->h_x_.data();
    }

    float* ParticleStates::host_y()
    {
        return this->h_y_.data();
    }

    float* ParticleStates::host_vx()
    {
        return this->h_vx_.data();
    }

    float* ParticleStates::host_vy()
    {
        return this->h_vy_.data();
    }

    float* ParticleStates::host_w()
    {
        return this->h_w_.data();
    }

    float* ParticleStates::host_h()
    {
        return this->h_h_.data();
    }

    float* ParticleStates::host_qw()
    {
        return this->h_qw_.data();
    }

    float* ParticleStates::host_qx()
    {
        return this->h_qx_.data();
    }

    float* ParticleStates::host_qy()
    {
        return this->h_qy_.data();
    }

    float* ParticleStates::host_qz()
    {
        return this->h_qz_.data();
    }

    float* ParticleStates::host_mean()
    {
        return this->h_mean_.data();
    }

    float* ParticleStates::host_cov()
    {
        return this->h_cov_.data();
    }

    float* ParticleStates::host_weights()
    {
        return this->h_weights_.data();
    }

    void ParticleStates::resize_(size_t num_particles)
    {
        this->d_x_.resize(num_particles);
        this->d_y_.resize(num_particles);
        this->h_x_.resize(num_particles);
        this->h_y_.resize(num_particles);
        this->d_vx_.resize(num_particles);
        this->d_vy_.resize(num_particles);
        this->h_vx_.resize(num_particles);
        this->h_vy_.resize(num_particles);
        this->d_w_.resize(num_particles);
        this->d_h_.resize(num_particles);
        this->h_w_.resize(num_particles);
        this->h_h_.resize(num_particles);
        this->d_qw_.resize(num_particles);
        this->d_qx_.resize(num_particles);
        this->d_qy_.resize(num_particles);
        this->d_qz_.resize(num_particles);
        this->h_qw_.resize(num_particles);
        this->h_qx_.resize(num_particles);
        this->h_qy_.resize(num_particles);
        this->h_qz_.resize(num_particles);
        this->d_mean_.resize(10);
        this->h_mean_.resize(10);
        this->d_cov_.resize(100);
        this->h_cov_.resize(100);
        this->d_weights_.resize(num_particles);
        this->h_weights_.resize(num_particles);
    }

} // namespace particle_filter