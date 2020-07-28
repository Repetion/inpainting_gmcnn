import argparse
import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, help='The name of dataset')
parser.add_argument('--test_path', type=str, help='The path of test dataset')
parser.add_argument('--result_num', type=int, default=0, help='The number of result (0 for all)')

def generate_mask(image_path, image_name, H, W):
    min_num_vertex = 4
    max_num_vertex = 12

    mean_angle = 2 * math.pi / 5
    angle_range = 2 * math.pi / 15
    min_width = 12
    max_width = 40
    average_radius = math.sqrt(H * H + W * W) / 8
    
    image = Image.open(image_path)
    image = image.convert('RGB')
    mask = Image.new('L', (W, H), 0)

    for _ in range(np.random.randint(1, 4)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []

        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(
                    2 * math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        vertex.append((int(np.random.randint(0, W)), int(np.random.randint(0, H))))

        for i in range(num_vertex):
            r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2), 0, 2 * average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, W)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, H)
            vertex.append((int(new_x), int(new_y)))

        draw_image = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw_image.line(vertex, fill=(255, 255, 255), width=width)
        draw_mask.line(vertex, fill=255, width=width)

        for v in vertex:
            draw_image.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=(255, 255, 255))
            draw_mask.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=255)

    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.normal() > 0:
        mask.transpose(Image.FLIP_TOP_BOTTOM)

    mask_path = '/home/dataset/freeform_mask/' + image_name + '_mask.png'
    mask.save(mask_path)

    return image, mask_path

if __name__ == "__main__":
    args = parser.parse_args()

    image_paths = []
    image_names = []
    match_masks = []

    for dirname, dirnames, filenames in os.walk(args.test_path):
        for filename in filenames:
            image_names.append(filename)
            image_paths.append(os.path.join(dirname, filename))

    image_names.sort()
    image_paths.sort()
    
    if args.result_num == 0:
        for image_name, image_path in zip(image_names, image_paths):
            image = Image.open(image_path)
            result, mask_name = generate_mask(image_path, image_name.split('.')[0], image.size[1], image.size[0])
            match_masks.append(mask_name)

            result_path = '/home/dataset/' + args.dataset_name + '/test_masked/masked_' + image_name
            result.save(result_path)
    else:
        for count in range(args.result_num):
            image = Image.open(image_paths[count])
            result, mask_name = generate_mask(image_paths[count], image_names[count].split('.')[0], image.size[1], image.size[0])
            match_masks.append(mask_name)

            result_path = '/home/dataset/' + args.dataset_name + '/test_masked/masked_' + image_names[count]
            result.save(result_path)
    
    image_names_path = '/home/jisukim/inpainting_gmcnn/flist/' + args.dataset_name + '/image_names_' + args.dataset_name + '.flist'
    f = open(image_names_path, 'w')
    f.write('\n'.join(image_names))
    f.close()

    image_paths_path = '/home/jisukim/inpainting_gmcnn/flist/' + args.dataset_name + '/image_paths_' + args.dataset_name + '.flist'
    f = open(image_paths_path, 'w')
    f.write('\n'.join(image_paths))
    f.close()

    match_masks_path = '/home/jisukim/inpainting_gmcnn/flist/' + args.dataset_name + '/match_masks_' + args.dataset_name + '.flist'
    f = open(match_masks_path, 'w')
    f.write('\n'.join(match_masks))
    f.close()