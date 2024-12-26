#pragma once

#include "STrack.h"
#include "tracker_interface.hpp" // Include the ITracker interface

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class BYTETracker : public ITracker // Inherit from ITracker
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	Scalar get_color(int idx);

	// Override the ITracker track method
    std::tuple<std::vector<cv::Rect>, std::vector<int>, std::vector<float>> track(const cv::Mat& frame, const std::vector<cv::Rect>& detections, const std::vector<int>& classIds, const std::vector<float>& scores) override;

private:
	// Original BYTETracker-specific method (now private)
    vector<STrack> update_(const vector<Object>& objects);
	vector<STrack*> joint_stracks(vector<STrack*> &tlista, vector<STrack> &tlistb);
	vector<STrack> joint_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);

	vector<STrack> sub_stracks(vector<STrack> &tlista, vector<STrack> &tlistb);
	void remove_duplicate_stracks(vector<STrack> &resa, vector<STrack> &resb, vector<STrack> &stracksa, vector<STrack> &stracksb);

	void linear_assignment(vector<vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		vector<vector<int> > &matches, vector<int> &unmatched_a, vector<int> &unmatched_b);
	vector<vector<float> > iou_distance(vector<STrack*> &atracks, vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	vector<vector<float> > iou_distance(vector<STrack> &atracks, vector<STrack> &btracks);
	vector<vector<float> > ious(vector<vector<float> > &atlbrs, vector<vector<float> > &btlbrs);

	double lapjv(const vector<vector<float> > &cost, vector<int> &rowsol, vector<int> &colsol, 
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	vector<STrack> tracked_stracks;
	vector<STrack> lost_stracks;
	vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};