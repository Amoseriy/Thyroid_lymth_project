import numpy as np
import nrrd
import os
import re


def process_nrrd_file(nrrd_path, label_class, label_info_by_person_and_z):
    print(f"开始提取{nrrd_path}")
    data, _ = nrrd.read(nrrd_path)
    unique_labels = np.unique(data)

    # 确定数据的维度
    if data.ndim == 4:
        z_slices = data.shape[3]
    elif data.ndim == 3:
        z_slices = data.shape[2]
    else:
        raise ValueError("Unexpected data dimensions")

    unique_z_values = np.unique(np.where(data > 0)[-1])
    unique_z_values = unique_z_values[unique_z_values < z_slices]
    # Extract person's name from the file name
    # print(nrrd_path)
    person_name = nrrd_path.split('/')[-2]
    print(person_name)
    for z_value in unique_z_values:
        label_info = label_info_by_person_and_z.get((person_name, z_value), [])
        z_slice_data = data[:, :, z_value]

        for label in unique_labels:
            if label != 0:  # Skip background label
                label_coords = np.where(z_slice_data == label)
                if label_coords[0].size > 0:
                    min_x, max_x = min(label_coords[0]), max(label_coords[0])
                    min_y, max_y = min(label_coords[1]), max(label_coords[1])
                    length, width = max_x - min_x + 1, max_y - min_y + 1
                    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
                    center_x_ratio, center_y_ratio = center_x / data.shape[0], center_y / data.shape[1]
                    length_ratio, width_ratio = length / data.shape[0], width / data.shape[1]
                    label_info.append([label_class, format(center_x_ratio, '.6f'), format(center_y_ratio, '.6f'),
                                       format(length_ratio, '.6f'), format(width_ratio, '.6f')])

        label_info_by_person_and_z[(person_name, z_value)] = label_info


def save_label_info(label_info_by_person_and_z, save_dir):
    for (person_name, z_value), infos in label_info_by_person_and_z.items():
        file_path = os.path.join(save_dir, f"{person_name}_{z_value}.txt")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(file_path, 'w') as f:
            for info in infos:
                f.write(' '.join(map(str, info)) + '\n')


def process_folder(folder_path, save_dir):
    label_info_by_person_and_z = {}
    class_labels = ['benign', 'malignant']
    # print(class_labels)
    file_path = folder_path.replace('\\', '/').replace('\\\\', '/')
    print(file_path)
    if file_path.endswith('.nrrd'):
        # print(file_name)
        label_class = 0 if '_B' in file_path else 1
        # print(label_class)
        nrrd_path = os.path.join(folder_path, file_path)
        process_nrrd_file(nrrd_path, label_class, label_info_by_person_and_z)

    save_label_info(label_info_by_person_and_z, save_dir)

    # Save classes.txt
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    classes_file_path = os.path.join(save_dir, 'classes.txt')
    with open(classes_file_path, 'w') as file:
        file.write('\n'.join(class_labels))


if __name__ == '__main__':
    # Set folder paths and start processing
    folder_path = "G:\Program\DATABASE\\2020\Xiang\ChenZhiQiang"
    save_dir = "./TXT/"
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.nrrd'):
                path = os.path.join(root, file_name)
                process_folder(path, save_dir)
