#import torch
#from diffusers import StableDiffusionPipeline
#
#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float32)
#
#pipe = pipe.to("cuda:3")
#prompt = "a photograph of an astronaut riding a horse"
#
#for i in range(10):
#    image = pipe(prompt, guidance_scale=i).images[0]
#    image.save(f"output_{i}.jpg")

import numpy as np
import glob
import pickle
import os

data_list = glob.glob('Data/processed/deepcad_subset_v2/val/*/*.npz')
for data_name in data_list:
    data = np.load(data_name)
    voxel_sdf = data['voxel_sdf']
    face_bounded_distance_field = data['face_bounded_distance_field']

    temp_data = {
        'voxel_sdf': voxel_sdf,
        'face_bounded_distance_field': face_bounded_distance_field
    }
    pickle.dump(temp_data, open('temp.pkl', 'wb'))

    data_id = data_name.split('/')[-2] + '-' + os.path.basename(data_name).split('.')[0]
    command = f'python brep_render.py --data_path temp.pkl --save_root temp_new/{data_id} '\
                f'--apply_nms --vis_each_face --vis_face_all --vis_face_only --vis_each_boundary'
    print(command)
    os.system(command)
