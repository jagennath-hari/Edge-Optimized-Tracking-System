#include "filter/particle_filter.hpp"

namespace particle_filter
{
    ParticleFilter::ParticleFilter(size_t num_particles, const float noise_x, const float noise_y, 
                                   const float noise_vx, const float noise_vy, 
                                   const float noise_w, const float noise_h, 
                                   const float noise_qw, const float noise_qx, const float noise_qy, const float noise_qz, 
                                   size_t top_percentage)
        : num_particles_(num_particles),
          top_percentage_(top_percentage),
          noise_x_(noise_x), noise_y_(noise_y), noise_vx_(noise_vx), noise_vy_(noise_vy),
          noise_w_(noise_w), noise_h_(noise_h), noise_qw_(noise_qw), noise_qx_(noise_qx), noise_qy_(noise_qy), noise_qz_(noise_qz)
    {
    }

    void ParticleFilter::initialize(int object_id, float init_x, float init_y, float init_vx, float init_vy, 
                                    float init_w, float init_h, float init_qw, float init_qx, float init_qy, float init_qz)
    {
        if (this->objects_.find(object_id) != this->objects_.end())
        {
            throw std::runtime_error("Object with ID " + std::to_string(object_id) + " is already being tracked.");
        }

        std::unique_ptr<ParticleStates> particles = std::make_unique<ParticleStates>(num_particles_);

        particle_filter_kernels::initParticles(
            particles->device_x(), particles->device_y(), 
            particles->device_vx(), particles->device_vy(), 
            particles->device_w(), particles->device_h(), 
            particles->device_qw(), particles->device_qx(), 
            particles->device_qy(), particles->device_qz(), 
            particles->device_weights(), 
            init_x, init_y, init_vx, init_vy, init_w, init_h, 
            init_qw, init_qx, init_qy, init_qz, 
            this->num_particles_, this->noise_x_, this->noise_y_, this->noise_vx_, this->noise_vy_, 
            this->noise_w_, this->noise_h_, this->noise_qw_, this->noise_qx_, this->noise_qy_, this->noise_qz_, static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

        ObjectState state = { std::move(particles) };
        this->objects_[object_id] = std::move(state);
    }

    void ParticleFilter::removeObject(int object_id)
    {
        this->objects_.erase(object_id);
    }

    void ParticleFilter::predict(int object_id)
    {
        // Find the object with the given ID
        std::unordered_map<int, ObjectState>::iterator it = this->objects_.find(object_id);
        if (it == this->objects_.end())
        {
            std::cerr << "Object " << object_id << " is not being tracked." << std::endl;
            return;
        }

        // Perform prediction for the specified object
        ObjectState& state = it->second;
        particle_filter_kernels::predict(
            state.particles->device_x(), state.particles->device_y(), 
            state.particles->device_vx(), state.particles->device_vy(), 
            state.particles->device_w(), state.particles->device_h(), 
            state.particles->device_qw(), state.particles->device_qx(), 
            state.particles->device_qy(), state.particles->device_qz(), 
            state.particles->device_weights(), state.particles->device_mean(), 
            state.particles->device_cov(), this->num_particles_, 
            this->noise_x_, this->noise_y_, 
            this->noise_w_, this->noise_h_, this->noise_qw_, this->noise_qx_, this->noise_qy_, this->noise_qz_);
    }

    void ParticleFilter::updateObject(int object_id, float obs_x, float obs_y, float obs_vx, float obs_vy, 
                                      float obs_w, float obs_h, float obs_qw, float obs_qx, float obs_qy, float obs_qz)
    {
        std::unordered_map<int, ObjectState>::iterator it = this->objects_.find(object_id);
        if (it == this->objects_.end())
        {
            std::cerr << "Object " << object_id << " is not being tracked." << std::endl;
            return;
        }

        ObjectState& state = it->second;
        particle_filter_kernels::updateParticles(
            state.particles->device_x(), state.particles->device_y(), 
            state.particles->device_vx(), state.particles->device_vy(), 
            state.particles->device_w(), state.particles->device_h(), 
            state.particles->device_qw(), state.particles->device_qx(), 
            state.particles->device_qy(), state.particles->device_qz(), 
            state.particles->device_weights(), obs_x, obs_y, obs_vx, obs_vy, 
            obs_w, obs_h, obs_qw, obs_qx, obs_qy, obs_qz, 
            this->noise_x_, this->noise_y_, this->noise_vx_, this->noise_vy_, 
            this->noise_w_, this->noise_h_, 
            this->num_particles_, static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
    }

    std::vector<float> ParticleFilter::getObjectState(int object_id) const
    {
        std::unordered_map<int, ObjectState>::const_iterator it = this->objects_.find(object_id);
        if (it == this->objects_.end())
        {
            throw std::runtime_error("Object " + std::to_string(object_id) + " is not being tracked.");
        }

        const ObjectState& state = it->second;
        return this->computeMeanState_(*state.particles);
    }

    std::vector<std::vector<float>> ParticleFilter::getObjectParticles(int object_id) const
    {
        std::unordered_map<int, ObjectState>::const_iterator it = this->objects_.find(object_id);
        if (it == this->objects_.end())
        {
            throw std::runtime_error("Object " + std::to_string(object_id) + " is not being tracked.");
        }

        const ObjectState& state = it->second;
        state.particles->download();

        std::vector<std::vector<float>> particles(state.particles->size(), std::vector<float>(10));
        float* host_x = state.particles->host_x();
        float* host_y = state.particles->host_y();
        float* host_vx = state.particles->host_vx();
        float* host_vy = state.particles->host_vy();
        float* host_w = state.particles->host_w();
        float* host_h = state.particles->host_h();
        float* host_qw = state.particles->host_qw();
        float* host_qx = state.particles->host_qx();
        float* host_qy = state.particles->host_qy();
        float* host_qz = state.particles->host_qz();

        for (size_t i = 0; i <state.particles->size(); ++i)
        {
            particles[i] = { host_x[i], host_y[i], host_vx[i], host_vy[i], host_w[i], host_h[i], host_qw[i], host_qx[i], host_qy[i], host_qz[i] };
        }
        
        return particles;
    }

    bool ParticleFilter::isTrackingObject(int object_id) const
    {
        return this->objects_.find(object_id) != this->objects_.end();
    }

    std::vector<float> ParticleFilter::computeMeanState_(particle_filter::ParticleStates& particles) const
    {
        particles.download();

        float* host_x = particles.host_x();
        float* host_y = particles.host_y();
        float* host_vx = particles.host_vx();
        float* host_vy = particles.host_vy();
        float* host_w = particles.host_w();
        float* host_h = particles.host_h();
        float* host_qw = particles.host_qw();
        float* host_qx = particles.host_qx();
        float* host_qy = particles.host_qy();
        float* host_qz = particles.host_qz();
        float* host_weights = particles.host_weights();

        size_t top_count = this->num_particles_ * this->top_percentage_ / 100;

        std::vector<size_t> indices(this->num_particles_);
        for (size_t i = 0; i < this->num_particles_; ++i)
        {
            indices[i] = i;
        }

        std::partial_sort(indices.begin(), indices.begin() + top_count, indices.end(),
            [&host_weights](size_t a, size_t b) { return host_weights[a] > host_weights[b]; });

        float mean_x = 0.0f, mean_y = 0.0f, mean_vx = 0.0f, mean_vy = 0.0f;
        float mean_w = 0.0f, mean_h = 0.0f, mean_qw = 0.0f, mean_qx = 0.0f, mean_qy = 0.0f, mean_qz = 0.0f;
        float total_weight = 0.0f;

        for (size_t i = 0; i < top_count; ++i)
        {
            size_t idx = indices[i];
            float weight = host_weights[idx];
            mean_x += host_x[idx] * weight;
            mean_y += host_y[idx] * weight;
            mean_vx += host_vx[idx] * weight;
            mean_vy += host_vy[idx] * weight;
            mean_w += host_w[idx] * weight;
            mean_h += host_h[idx] * weight;
            mean_qw += host_qw[idx] * weight;
            mean_qx += host_qx[idx] * weight;
            mean_qy += host_qy[idx] * weight;
            mean_qz += host_qz[idx] * weight;
            total_weight += weight;
        }

        if (total_weight > 0.0f)
        {
            mean_x /= total_weight;
            mean_y /= total_weight;
            mean_vx /= total_weight;
            mean_vy /= total_weight;
            mean_w /= total_weight;
            mean_h /= total_weight;
            mean_qw /= total_weight;
            mean_qx /= total_weight;
            mean_qy /= total_weight;
            mean_qz /= total_weight;
        }

        return { mean_x, mean_y, mean_vx, mean_vy, mean_w, mean_h, mean_qw, mean_qx, mean_qy, mean_qz };
    }
}
