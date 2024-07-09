import json
import os
import argparse

def load_rtmpose_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def convert_to_pose2sim_format(frame_data):
    pose2sim_frame = {
        "version": 1.3,
        "people": []
    }
    
    for instance in frame_data['instances']:
        keypoints = instance['keypoints']
        scores = instance['keypoint_scores']
        
        pose_keypoints_2d = []
        for idx, point in enumerate(keypoints):
            pose_keypoints_2d.extend(point)
            pose_keypoints_2d.append(scores[idx])
        
        person_data = {
            "person_id": [-1],
            "pose_keypoints_2d": pose_keypoints_2d,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }
        
        pose2sim_frame["people"].append(person_data)
    
    return pose2sim_frame

def save_pose2sim_json(data, frame_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, f"frame_{frame_id}.json")
    with open(output_filepath, 'w') as f:
        json.dump(data, f, indent=4)

def convert_rtmpose_to_pose2sim(input_filepath, output_dir):
    rtmpose_data = load_rtmpose_json(input_filepath)
    frames = rtmpose_data['instance_info']
    for frame in frames:
        frame_id = frame['frame_id']
        pose2sim_frame = convert_to_pose2sim_format(frame)
        save_pose2sim_json(pose2sim_frame, frame_id, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Convert RTMPose JSON to Pose2Sim JSON format.')
    parser.add_argument('-i', '--input', required=True, help='Path to the input RTMPose JSON file.')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory where Pose2Sim JSON files will be saved.')
    args = parser.parse_args()
    
    input_filepath = args.input
    output_dir = args.output
    
    convert_rtmpose_to_pose2sim(input_filepath, output_dir)

if __name__ == '__main__':
    main()
