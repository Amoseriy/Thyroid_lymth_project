import os
import SimpleITK as sitk
import numpy as np

root = "./DATABASE"
lymph_nodes_num = 0
slices_num = 0
for year in os.listdir(root):
    year_path = os.path.join(root, year)
    # print(year_path)
    for researchers in os.listdir(year_path):
        researcher_path = os.path.join(year_path, researchers)
        # print(researcher_path)
        for patient in os.listdir(researcher_path):
            # path_list = []
            patient_path = os.path.join(researcher_path, patient)
            # print(patient_path)
            for file in os.listdir(patient_path):
                file_path = os.path.join(patient_path, file)
                # path_list.append(file_path)
                if file.endswith(".seg.nrrd"):
                    # 读取seg.nrrd文件
                    seg_image = sitk.ReadImage(file_path)

                    # 将SimpleITK图像转换为numpy数组
                    seg_array = sitk.GetArrayFromImage(seg_image)
                    # 获取标签值（排除背景标签0）
                    labels = np.unique(seg_array)
                    labels = labels[labels > 0]  # 移除背景标签0
                    lymph_nodes_num += len(labels)
                    print(f"{file_path}提取完毕")
                    # 获取每个标签的层面数量
                    label_slices = {}
                    for label in labels:
                        # 对于每个标签，找到在哪些层面出现
                        slices = np.any(seg_array == label, axis=(1, 2))  # 沿Z轴方向，判断每个slice是否有该标签
                        slice_indices = np.where(slices)[0]  # 返回标签所在的层面索引
                        label_slices[label] = len(slice_indices)
                        slices_num += len(slice_indices)

print(f"共有{lymph_nodes_num}个淋巴结")
print(f"共有{slices_num}个层面")


