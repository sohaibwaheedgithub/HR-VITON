# Import libraries

import os
import sys
import json
import shutil
import random
import subprocess
from tqdm import tqdm
from glob import glob
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision




IMAGE_DIR = '/notebooks/Parent/dataset/train/image'

'''================ Train and Test pairs text files Generation ================'''

im_paths = glob(f'{IMAGE_DIR}/*.jpg')
im_names = [os.path.basename(path) for path in im_paths]

with open('/notebooks/Parent/dataset/train_pairs.txt', 'w') as f:    
    for im_name in im_names:
        for c_name in im_names:
            if im_name != c_name:
                f.write(f'{im_name} {c_name}\n')
    

c_names = im_names.copy()
random.shuffle(c_names)
with open('/notebooks/Parent/dataset/test_pairs.txt', 'w') as f:
    for im_name, c_name in zip(im_names, c_names):
        f.write(f"{im_name} {c_name}\n")



'''================ DensePose Generation ================'''

densepose_dir = '/notebooks/Parent/dataset/train/image-densepose'
if os.path.exists(densepose_dir):
    shutil.rmtree(densepose_dir)
    os.mkdir(densepose_dir)
    print("image-densepose directory has been created")
    
command = f'''
source /notebooks/Parent/miniconda/etc/profile.d/conda.sh && \
conda activate densepose && \
python /notebooks/Parent/detectron2/projects/DensePose/mod_apply_net.py show \
    /notebooks/Parent/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml \
        /notebooks/Parent/detectron2/projects/DensePose/model/model_final_162be9.pkl \
            {IMAGE_DIR} dp_segm --output \
                {densepose_dir}/output.jpg
'''


print("Generating DensePose")
process = subprocess.run(command, shell=True, executable="/bin/bash")

if process.returncode == 0:
    print("DensePose has been generated successfully")
else:
    print("DensePose generation failed")
    sys.exit(0)
    


'''================ Parse, Cloth And Cloth-Mask Generation ================''' 

cloth_dir = '/notebooks/Parent/dataset/train/cloth'
cloth_mask_dir = '/notebooks/Parent/dataset/train/cloth-mask'
img_parse_dir = '/notebooks/Parent/dataset/train/image-parse-v3'


if os.path.exists(cloth_dir):
    shutil.rmtree(cloth_dir)
    os.mkdir(cloth_dir)
    print("cloth directory has been created")
    
    
if os.path.exists(cloth_mask_dir):
    shutil.rmtree(cloth_mask_dir)
    os.mkdir(cloth_mask_dir)
    print("cloth_mask directory has been created")
    
    
if os.path.exists(img_parse_dir):
    shutil.rmtree(img_parse_dir)
    os.mkdir(img_parse_dir)
    print("image-parse-v3 directory has been created")
    
    
command = f'''
source /notebooks/Parent/miniconda/etc/profile.d/conda.sh && \
conda activate jppnet && \
python /notebooks/Parent/LIP-JPPNet-TensorFlow/mod_evaluate_parsing_JPPNet-s2.py
'''

print("Generating Image-Parse, Cloth and Cloth-Masks")
process = subprocess.run(command, shell=True, executable='/bin/bash')

if process.returncode == 0:
    print("Parse, Cloth and Cloth-mask has been generated successfully")
else:
    print("Parse, Cloth and Cloth-mask generation failed")
    sys.exit(0)
    
    
   
'''================ Parse-Agnostic Generation ================'''

parse_agnostic_dir = '/notebooks/Parent/dataset/train/image-parse-agnostic-v3.2'
    
if os.path.exists(parse_agnostic_dir):
    shutil.rmtree(parse_agnostic_dir)
    os.mkdir(parse_agnostic_dir)
    print("image-parse-agnostic-v3 directory has been created")
    
command = f'''
source /notebooks/Parent/miniconda/etc/profile.d/conda.sh && \
conda activate hr_viton && \
python /notebooks/Parent/HR-VITON/get_parse_agnostic.py --data_path /notebooks/Parent/dataset/train --output_path {parse_agnostic_dir}
'''

process = subprocess.run(command, shell=True, executable='/bin/bash')

if process.returncode == 0:
    print('Parse-Agnostic has been generated successfully')
else:
    print('Parse-Agnostic generation failed')




'''================ Openpose Image And Keypoints Generation ================'''

# Fake Openpose Images Generation
openpose_img_dir = '/notebooks/Parent/dataset/train/openpose_img'
openpose_json_dir = '/notebooks/Parent/dataset/train/openpose_json'

if os.path.exists(openpose_img_dir):
    shutil.rmtree(openpose_img_dir)
    os.mkdir(openpose_img_dir)
    print("openpose_img directory has been created")
    

if os.path.exists(openpose_json_dir):
    shutil.rmtree(openpose_json_dir)
    os.mkdir(openpose_json_dir)
    print("openpose_json directory has been created")


img = Image.open(os.path.join(IMAGE_DIR, im_names[0]))

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


def create_keypoints_json(rgb_image, body_detection_result, hands_detection_result, image_name):
  pose_landmarks_list = body_detection_result.pose_landmarks
  hand_landmarks_list = hands_detection_result.hand_landmarks
  handedness_list = hands_detection_result.handedness

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    kp_1_x = ((pose_landmarks[11].x * rgb_image.shape[1]) + (pose_landmarks[12].x * rgb_image.shape[1])) / 2
    kp_1_y = ((pose_landmarks[11].y * rgb_image.shape[0]) + (pose_landmarks[12].y * rgb_image.shape[0])) / 2
    kp_1_v = (pose_landmarks[11].visibility + pose_landmarks[12].visibility) / 2

    kp_8_x = ((pose_landmarks[23].x * rgb_image.shape[1]) + (pose_landmarks[24].x * rgb_image.shape[1])) / 2
    kp_8_y = ((pose_landmarks[23].y * rgb_image.shape[0]) + (pose_landmarks[24].y * rgb_image.shape[0])) / 2
    kp_8_v = (pose_landmarks[23].visibility + pose_landmarks[24].visibility) / 2

    pose_keypoints_2d = [
    pose_landmarks[0].x * rgb_image.shape[1], pose_landmarks[0].y * rgb_image.shape[0], pose_landmarks[0].visibility,
    kp_1_x, kp_1_y, kp_1_v,
    pose_landmarks[12].x * rgb_image.shape[1], pose_landmarks[12].y * rgb_image.shape[0], pose_landmarks[12].visibility,
    pose_landmarks[14].x * rgb_image.shape[1], pose_landmarks[14].y * rgb_image.shape[0], pose_landmarks[14].visibility,
    pose_landmarks[16].x * rgb_image.shape[1], pose_landmarks[16].y * rgb_image.shape[0], pose_landmarks[16].visibility,
    pose_landmarks[11].x * rgb_image.shape[1], pose_landmarks[11].y * rgb_image.shape[0], pose_landmarks[11].visibility,
    pose_landmarks[13].x * rgb_image.shape[1], pose_landmarks[13].y * rgb_image.shape[0], pose_landmarks[13].visibility,
    pose_landmarks[15].x * rgb_image.shape[1], pose_landmarks[15].y * rgb_image.shape[0], pose_landmarks[15].visibility,
    kp_8_x, kp_8_y, kp_8_v,
    pose_landmarks[24].x * rgb_image.shape[1], pose_landmarks[24].y * rgb_image.shape[0], pose_landmarks[24].visibility,
    pose_landmarks[26].x * rgb_image.shape[1], pose_landmarks[26].y * rgb_image.shape[0], pose_landmarks[26].visibility,
    pose_landmarks[28].x * rgb_image.shape[1], pose_landmarks[28].y * rgb_image.shape[0], pose_landmarks[28].visibility,
    pose_landmarks[23].x * rgb_image.shape[1], pose_landmarks[23].y * rgb_image.shape[0], pose_landmarks[23].visibility,
    pose_landmarks[25].x * rgb_image.shape[1], pose_landmarks[25].y * rgb_image.shape[0], pose_landmarks[25].visibility,
    pose_landmarks[27].x * rgb_image.shape[1], pose_landmarks[27].y * rgb_image.shape[0], pose_landmarks[27].visibility,
    pose_landmarks[5].x * rgb_image.shape[1], pose_landmarks[5].y * rgb_image.shape[0], pose_landmarks[5].visibility,
    pose_landmarks[2].x * rgb_image.shape[1], pose_landmarks[2].y * rgb_image.shape[0], pose_landmarks[2].visibility,
    pose_landmarks[8].x * rgb_image.shape[1], pose_landmarks[8].y * rgb_image.shape[0], pose_landmarks[8].visibility,
    pose_landmarks[7].x * rgb_image.shape[1], pose_landmarks[7].y * rgb_image.shape[0], pose_landmarks[7].visibility,
    pose_landmarks[29].x * rgb_image.shape[1], pose_landmarks[29].y * rgb_image.shape[0], pose_landmarks[29].visibility,
    pose_landmarks[31].x * rgb_image.shape[1], pose_landmarks[31].y * rgb_image.shape[0], pose_landmarks[31].visibility,
    pose_landmarks[27].x * rgb_image.shape[1], pose_landmarks[27].y * rgb_image.shape[0], pose_landmarks[27].visibility,
    pose_landmarks[30].x * rgb_image.shape[1], pose_landmarks[30].y * rgb_image.shape[0], pose_landmarks[30].visibility,
    pose_landmarks[32].x * rgb_image.shape[1], pose_landmarks[32].y * rgb_image.shape[0], pose_landmarks[32].visibility,
    pose_landmarks[28].x * rgb_image.shape[1], pose_landmarks[28].y * rgb_image.shape[0], pose_landmarks[28].visibility
    ]


  # Loop through the detected hands to visualize.

  if len(hand_landmarks_list) != 0:

    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      hand_right_keypoints_2d = []
      hand_left_keypoints_2d = []
      if handedness[0].category_name == 'Right':
        for lmk in hand_landmarks:
          hand_right_keypoints_2d.append(lmk.x * rgb_image.shape[1])
          hand_right_keypoints_2d.append(lmk.y * rgb_image.shape[0])
          hand_right_keypoints_2d.append(lmk.visibility)
      if handedness[0].category_name == 'Left':
        for lmk in hand_landmarks:
          hand_left_keypoints_2d.append(lmk.x * rgb_image.shape[1])
          hand_left_keypoints_2d.append(lmk.y * rgb_image.shape[0])
          hand_left_keypoints_2d.append(lmk.visibility)

  else:
    hand_right_keypoints_2d = []
    hand_left_keypoints_2d = []



  json_dict = {
        "version":1.3,
        "people":[
            {
                "person_id":[-1],
                "pose_keypoints_2d": pose_keypoints_2d,
                "face_keypoints_2d":[],
                "hand_left_keypoints_2d": hand_left_keypoints_2d,
                "hand_right_keypoints_2d": hand_right_keypoints_2d,
                "pose_keypoints_3d":[],
                "face_keypoints_3d":[],
                "hand_left_keypoints_3d":[],
                "hand_right_keypoints_3d":[]
            }
        ]
  }

  json_obj = json.dumps(json_dict)

  with open(f'{openpose_json_dir}/{image_name}_keypoints.json', 'w') as f:
    f.write(json_obj)
    

    
print("Generating Openpose Jsons And Images")    
for im_name in tqdm(im_names):
    img.save(os.path.join(openpose_img_dir, f'{im_name[:-4]}_rendered.png'))
    
    image = mp.Image.create_from_file(os.path.join(IMAGE_DIR, im_name))

    # For Body Landmarks
    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 4: Detect pose landmarks from the input image.
    body_detection_result = detector.detect(image)

    # For Hands_landmarks
    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 4: Detect hand landmarks from the input image.
    hands_detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    create_keypoints_json(image.numpy_view(), body_detection_result, hands_detection_result, im_name[:-4])

print("Openpose Jsons And Images Generation Completed")    
    
    
# Copying all train data into test directory

test_dir = '/notebooks/Parent/dataset/test'

if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
    os.mkdir(test_dir)
    print("Test directory has been created")

command = f'cp -r /notebooks/Parent/dataset/train/* {test_dir}/'

exit_code = os.system(command)

if exit_code == 0:
    print("Train sub-directories has been successfully copied into Test")
else:
    print("Train sub-directories copy into Test failed")