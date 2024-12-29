import math
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from data_loader import DataLoader
from yolox.tracker.byte_tracker import BYTETracker
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class BYTETrackerArgs:
    track_thresh: float = 0.01
    track_buffer: int = 50
    match_thresh: float = 0.99
    aspect_ratio_thresh: float = 200.0
    min_box_area: float = 0.05
    mot20: bool = True

class ParticleFilter:
    def __init__(self, num_particles=100, noise_std=(5.0, 5.0, 2.0, 2.0, 2.0, 2.0, 0.1)):
        """
        Initialize the particle filter.
        """
        self.num_particles = num_particles
        self.noise_std = np.array(noise_std)
        self.particles = defaultdict(dict)  # {track_id: {"particles": ..., "weights": ...}}

    def initialize_particles(self, track_id, x, y, vx, vy, w, h, quaternion):
        """
        Initialize particles for a new object track within the bounding box.
        """
        particles = np.zeros((self.num_particles, 10))  # x, y, vx, vy, w, h, q_w, q_x, q_y, q_z
        particles[:, 0] = np.random.uniform(x - w / 2, x + w / 2, self.num_particles)  # x
        particles[:, 1] = np.random.uniform(y - h / 2, y + h / 2, self.num_particles)  # y
        particles[:, 2] = np.random.normal(vx, self.noise_std[2], self.num_particles)  # vx
        particles[:, 3] = np.random.normal(vy, self.noise_std[3], self.num_particles)  # vy
        particles[:, 4] = np.random.normal(w, self.noise_std[4], self.num_particles)  # w
        particles[:, 5] = np.random.normal(h, self.noise_std[5], self.num_particles)  # h

        quaternion = np.array(quaternion)  # Ensure quaternion is an array
        if quaternion.shape != (4,):
            raise ValueError("Quaternion must have 4 components: [q_w, q_x, q_y, q_z]")
        particles[:, 6:10] = np.tile(quaternion, (self.num_particles, 1))

        weights = np.ones(self.num_particles) / self.num_particles
        self.particles[track_id] = {"particles": particles, "weights": weights}

    def unscented_transform(self, mean, covariance, alpha=1e-2, beta=2, kappa=0):
        """
        Generate sigma points and weights using the unscented transform for a 10-state system.
        """
        n = mean.shape[0]  # Dimensionality of the state vector
        lambda_ = alpha**2 * (n + kappa) - n

        # Weights for mean and covariance
        weights_mean = np.full(2 * n + 1, 0.5 / (n + lambda_))
        weights_covariance = np.full(2 * n + 1, 0.5 / (n + lambda_))
        weights_mean[0] = lambda_ / (n + lambda_)
        weights_covariance[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

        # Compute sigma points
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        sqrt_matrix = np.linalg.cholesky((n + lambda_) * covariance)
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_matrix[i]
            sigma_points[n + i + 1] = mean - sqrt_matrix[i]

        # Ensure quaternions (states 6–9) are normalized
        for i in range(2 * n + 1):
            quat = sigma_points[i, 6:10]
            sigma_points[i, 6:10] = quat / np.linalg.norm(quat)

        return sigma_points, weights_mean, weights_covariance

    def predict(self, track_id):
        """
        Predict the next state using the unscented transform for a given track ID.
        """
        if track_id not in self.particles:
            return

        particles = self.particles[track_id]["particles"]
        weights = self.particles[track_id]["weights"]

        # Compute the mean and covariance of the current particles
        mean = np.average(particles, axis=0, weights=weights)
        covariance = np.cov(particles.T, aweights=weights)

        # Generate sigma points
        sigma_points, weights_mean, weights_covariance = self.unscented_transform(mean, covariance)

        # Propagate sigma points through the motion model
        propagated_sigma_points = np.zeros_like(sigma_points)
        for i, sigma in enumerate(sigma_points):
            propagated_sigma_points[i, :6] = sigma[:6]  # Linear states
            # Quaternion propagation: Add noise to rotation
            delta_theta = np.random.normal(0, self.noise_std[6])
            delta_quat = R.from_euler('z', delta_theta).as_quat()
            propagated_quat = (R.from_quat(sigma[6:10]) * R.from_quat(delta_quat)).as_quat()
            propagated_sigma_points[i, 6:10] = propagated_quat

        # Calculate the predicted mean and covariance
        predicted_mean = np.dot(weights_mean, propagated_sigma_points)
        predicted_covariance = sum(
            weights_covariance[j] * np.outer(propagated_sigma_points[j] - predicted_mean,
                                             propagated_sigma_points[j] - predicted_mean)
            for j in range(propagated_sigma_points.shape[0])
        )

        # Resample particles from the predicted distribution
        noise = np.random.multivariate_normal(predicted_mean[:6], predicted_covariance[:6, :6], self.num_particles)
        self.particles[track_id]["particles"][:, :6] = noise

        # Normalize quaternions
        for i in range(self.num_particles):
            quat = propagated_sigma_points[0, 6:10]
            self.particles[track_id]["particles"][i, 6:10] = quat / np.linalg.norm(quat)

    def update(self, track_id, x, y, vx, vy, w, h, quaternion):
        """
        Update the particles based on a new observation.
        """
        if track_id not in self.particles:
            return

        particles = self.particles[track_id]["particles"]
        weights = self.particles[track_id]["weights"]

        # Compute weights based on proximity (linear states)
        distances = np.linalg.norm(particles[:, :2] - np.array([x, y]), axis=1)
        weights = np.exp(-distances / 10)  # Exponential weighting for proximity
        weights += 1e-10  # Avoid division by zero
        weights /= np.sum(weights)  # Normalize
        self.particles[track_id]["weights"] = weights

        # Update particle states with weighted blending
        particles[:, 0] = 0.5 * particles[:, 0] + 0.5 * x + np.random.normal(0, self.noise_std[0], self.num_particles)
        particles[:, 1] = 0.5 * particles[:, 1] + 0.5 * y + np.random.normal(0, self.noise_std[1], self.num_particles)
        particles[:, 2] = 0.5 * particles[:, 2] + 0.5 * vx + np.random.normal(0, self.noise_std[2], self.num_particles)
        particles[:, 3] = 0.5 * particles[:, 3] + 0.5 * vy + np.random.normal(0, self.noise_std[3], self.num_particles)
        particles[:, 4] = 0.5 * particles[:, 4] + 0.5 * w + np.random.normal(0, self.noise_std[4], self.num_particles)
        particles[:, 5] = 0.5 * particles[:, 5] + 0.5 * h + np.random.normal(0, self.noise_std[5], self.num_particles)

        # Update quaternion using SLERP
        observation_quat = np.array(quaternion)
        for i in range(self.num_particles):
            current_quat = particles[i, 6:10]
            current_rotation = R.from_quat(current_quat)

            # Smoothly interpolate between the current and observed quaternion
            updated_quat = Slerp([0, 1], R.from_quat([current_quat, observation_quat]))(0.5).as_quat()

            # Normalize quaternion and update particle
            particles[i, 6:10] = updated_quat / np.linalg.norm(updated_quat)

        # Resample particles after updating weights
        self.resample(track_id)

    def resample(self, track_id):
        """
        Resample particles based on their weights.
        """
        particles = self.particles[track_id]["particles"]
        weights = self.particles[track_id]["weights"]

        # Resample indices based on weights
        indices = np.random.choice(len(particles), size=len(particles), p=weights)

        # Resample particles and reset weights
        self.particles[track_id]["particles"] = particles[indices]
        self.particles[track_id]["weights"] = np.ones(len(particles)) / len(particles)


    def estimate(self, track_id):
        """
        Estimate the state of the object from the particles.
        """
        if track_id not in self.particles:
            return None

        particles = self.particles[track_id]["particles"]
        weights = self.particles[track_id]["weights"]

        # Weighted average for estimation
        state = np.average(particles, axis=0, weights=weights)
        return state


class LearningTracker:
    def __init__(self, yolo_path):
        self.model = YOLO(yolo_path).to('cuda:0')
        self.data = DataLoader('/home/hari/Downloads/SportsMOT_example/dataset/train/v_gQNyhv8y0QY_c013/img1/')
        self.byte_tracker = BYTETracker(BYTETrackerArgs())
        self.particle_filter = ParticleFilter()
        self.previous_states = {}
    
    def predict_future_positions(self, x, y, vx, vy, num_frames=5, delta_time=1.0):
        """
        Predict future positions based on current velocity and position.
        Args:
            x, y: Current position.
            vx, vy: Current velocity.
            num_frames: Number of future frames to predict.
            delta_time: Time step per frame.
        Returns:
            List of predicted positions [(x1, y1), (x2, y2), ...].
        """
        predictions = []
        for frame in range(1, num_frames + 1):
            future_x = x + vx * delta_time * frame
            future_y = y + vy * delta_time * frame
            predictions.append((future_x, future_y))
        return predictions


    def runTracker(self, output_video_path="output_tracking.avi", fps=30):
        # Get the first frame's shape for video writer initialization
        first_frame, _ = next(self.data.run())
        frame_height, frame_width = first_frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use 'XVID' or another codec
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


        for curr_image, curr_annotations in self.data.run():
            results = self.model(curr_image, stream=True, imgsz=736, verbose=False)
            all_boxes = []

            for result in results:
                boxes = result.boxes
                data = boxes.data
                for box in data:
                    x1, y1, x2, y2, confidence, cls_id = box.cpu().numpy()
                    all_boxes.append([x1, y1, x2, y2, confidence])

            all_boxes_matrix = np.array(all_boxes)
            tracks = self.byte_tracker.update(all_boxes_matrix, curr_image.shape, curr_image.shape)

            for t in tracks:
                tlwh = t.tlwh
                tid = t.track_id

                x, y = tlwh[0] + tlwh[2] / 2, tlwh[1] + tlwh[3] / 2
                w, h = tlwh[2], tlwh[3]

                if tid in self.previous_states:
                    prev_x, prev_y = self.previous_states[tid]
                    vx, vy = x - prev_x, y - prev_y

                    # Compute quaternion from velocity direction
                    theta = math.atan2(vy, vx)
                    current_quaternion = R.from_euler('z', theta).as_quat()
                else:
                    vx, vy = 0, 0
                    current_quaternion = np.array([0, 0, 0, 1])  # Default quaternion (no rotation)

                self.previous_states[tid] = (x, y)

                if tid not in self.particle_filter.particles:
                    self.particle_filter.initialize_particles(tid, x, y, vx, vy, w, h, current_quaternion)
                else:
                    self.particle_filter.update(tid, x, y, vx, vy, w, h, current_quaternion)

                state = self.particle_filter.estimate(tid)
                x, y, _, _, w, h, q_w, q_x, q_y, q_z = state

                # Visualize all particles for the track
                for particle in self.particle_filter.particles[tid]["particles"]:
                    px, py = int(particle[0]), int(particle[1])
                    cv2.circle(curr_image, (px, py), 1, (0, 255, 255), -1)  # Yellow particles

                # Draw bounding box for the estimated state
                cv2.rectangle(curr_image, (int(x - w / 2), int(y - h / 2)), 
                            (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                cv2.putText(curr_image, f"ID: {tid}", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Predict and visualize future positions
                predictions = self.predict_future_positions(x, y, vx, vy, num_frames=5)
                for i, (px, py) in enumerate(predictions):
                    cv2.circle(curr_image, (int(px), int(py)), 3, (255, 255 - i * 40, i * 40), -1)  # Gradient for dots

                # Visualize quaternion rotation as an arrow
                rotation_vector = R.from_quat([q_w, q_x, q_y, q_z]).apply([1, 0, 0])  # Quaternion rotation
                arrow_end = (int(x + rotation_vector[0] * 50), int(y + rotation_vector[1] * 50))
                cv2.arrowedLine(curr_image, (int(x), int(y)), arrow_end, (0, 0, 255), 2, tipLength=0.3)

            # Write frame to video
            video_writer.write(curr_image)

        # Release video writer and close OpenCV windows
        video_writer.release()




def main():
    tracker = LearningTracker("/home/hari/particle_tracker/yolov11/runs/detect/train9/weights/best.pt")
    tracker.runTracker()

if __name__ == "__main__":
    main()