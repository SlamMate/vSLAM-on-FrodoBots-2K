# vSLAM-on-FrodoBots-2K
 FrodoBots-2K is a very cool dataset, We'll put some details on how to use it for visual SLAM.
# 1. Download the dataset
https://huggingface.co/datasets/frodobots/FrodoBots-2K
# 2. Calibration
According to my experiment, the approximate Calibration param is as the file Robot_Zero.yaml. We buy the robot, but it will take some time to deliver.
# 3. Merge video frames
The video frames in FrodoBots-2K are discrete, you need to merge them into a longer video.
# 4. Change your CmakeList.txt to compile the new Calibration model
```cmake
add_executable(Robot_zero
        Examples/Monocular/Robot_zero.cc)
target_link_libraries(Robot_zero ${PROJECT_NAME})
```
# 5. Run it in ORBSLAM3!
```bash
./Examples/Monocular/Robot_Zero Vocabulary/ORBvoc.txt Examples/Monocular/Robot_zero.yaml /home/zhangqi/Downloads/output_rides_21/ride_38222_20240501013650
```
