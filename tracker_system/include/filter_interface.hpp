#pragma once

#include <vector>

// Abstract base class for state estimation filters
class IFilter 
{
public:
    virtual ~IFilter() = default;

    // Initialize the filter for a tracked object ID
    virtual void initialize(int track_id, 
                            float init_x, float init_y, float init_vx, float init_vy,
                            float init_w, float init_h, float init_qw, float init_qx, 
                            float init_qy, float init_qz) = 0;

    // Prediction step for all particles
    virtual void predict(int track_id) = 0;

    // Update the state of the tracked object based on an observation
    virtual void updateObject(int track_id, 
                        float observed_x, float observed_y, float observed_vx, float observed_vy,
                        float observed_w, float observed_h, 
                        float observed_qw, float observed_qx, float observed_qy, float observed_qz) = 0;

    // Function to see if the object is tracking
    virtual bool isTrackingObject(int object_id) const = 0;

    // Function to remove the object ID
    virtual void removeObject(int object_id) = 0;

    // Function to get object state
    virtual std::vector<float> getObjectState(int object_id) const = 0;

    // Function to get the particles
    virtual std::vector<std::vector<float>> getObjectParticles(int object_id) const = 0;


};
