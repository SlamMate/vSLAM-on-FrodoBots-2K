# vSLAM-on-FrodoBots-2K
 FrodoBots-2K is a very cool dataset, We'll put some details on how to use it for visual SLAM.
## 1. SLAM setup(use the orbslam3 as an example)
### 1.1 Download the dataset
https://huggingface.co/datasets/frodobots/FrodoBots-2K
### 1.2 Calibration
According to my experiment, the approximate Calibration param is as the file Robot_Zero.yaml. We buy the robot, but it will take some time to deliver.
### 1.3 Merge video frames
The video frames in FrodoBots-2K are discrete, you need to merge them into a longer video.
The merge_ts_files.sh would help you!!
```bash
mv merge_ts_file.sh /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650
chmod +x merge_ts_file.sh
./merge_ts_file.sh
```
The sequence is ready to use!!!
### 1.4 Change your CmakeList.txt to compile the new Calibration model
```cmake
add_executable(Robot_zero
        Examples/Monocular/Robot_zero.cc)
target_link_libraries(Robot_zero ${PROJECT_NAME})
```
### 1.5 Run it in ORBSLAM3!
```bash
./Examples/Monocular/Robot_Zero Vocabulary/ORBvoc.txt Examples/Monocular/Robot_zero.yaml /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650
```
![Running in ORBSLAM3](images/example_image0.png)
## 2. Object Detection(Use the YOLOX as an example)
### 2.1 Merge video frames
The video frames in FrodoBots-2K are discrete, you need to merge them into a longer video.
The merge_ts_files.sh would help you!!
```bash
mv merge_ts_file.sh /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650
chmod +x merge_ts_file.sh
./merge_ts_file.sh
```
### 2.2 Produce the object detect the result of the merged video
Pls, download the file run_video.py in the rep
```bash
python run_video.py video -f /home/zhangqi/Documents/Library/YOLOX/exps/default/yolox_s.py -c /home/zhangqi/Documents/Library/YOLOX/yolox_s.pth --path /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650/recordings/rgb.ts --save_result
```
### 2.3 Get the video from ./YOLOX_outputs/yolox_s/vis_res/2024_07_07_21_30_59
![Running in YOLOX](images/example_image1.png)
## 3. Depth Estimation(Use the Lite-Mono as an example)
### 3.1 Merge video frames
The video frames in FrodoBots-2K are discrete, you need to merge them into a longer video.
The merge_ts_files.sh would help you!!
```bash
mv merge_ts_file.sh /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650
chmod +x merge_ts_file.sh
./merge_ts_file.sh
```
### 3.2 Estimating depth according to the result of the merged video
Pls, download the file video_depth_prediction.py in the rep.
And move the file to LiteMono/
The model is 1024 x 320
```bash
python video_depth_prediction.py --video_path /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650/recordings/rgb.ts --output_path output_video_depth.avi --load_weights_folder /home/zhangqi/Documents/Library/Lite-Mono/pretrained_model --model lite-mono8m
```
### 3.3 Get the video from LiteMono/
![Running in LiteMono](images/example_image2.png)
