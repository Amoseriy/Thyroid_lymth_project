#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author: Amoser @date: 2024/6/20 下午8:27


def get_zip_path(imgs, masks):
    zip_path = {}
    for jpg in imgs:
        tem_list = []
        parts = jpg.split("_", 2)
        for string in ["_L_B", "_R_B", "_L_M", "_L_B"]:
            matcher = f'{parts[0]}_{parts[1]}{string}_{parts[2]}'
            matcher = matcher.replace("jpg", "png")
            if matcher in masks:
                tem_list.append(matcher)
                # print(matcher)
        zip_path[jpg] = tem_list
    tem_keys = []
    for i in zip_path.keys():
        if len(zip_path[i]) == 0:
            tem_keys.append(i)
    for k in tem_keys:
        del zip_path[k]
    return zip_path


if __name__ == '__main__':
    with open("imgs.txt", "r") as f:
        jpg_list = [img.strip() for img in f.readlines()]

    with open("masks.txt", "r") as f:
        png_list = [mask.strip() for mask in f.readlines()]
    zip = get_zip_path(jpg_list, png_list)
    print(len(zip))
