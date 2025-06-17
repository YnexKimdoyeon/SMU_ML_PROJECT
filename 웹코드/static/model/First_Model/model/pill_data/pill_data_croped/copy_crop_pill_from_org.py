import cv2

from get_cli_args import get_cli_args
from pathlib import Path
import shutil
import utils
import numpy as np


list_dir_pill_org = [
    # r'G:\경구약제 DB\0.단일 경구약제 1000종\경구약제 1000종 추가 DB',
    # r'G:\경구약제 DB\0.단일 경구약제 1000종\경구약제 1000종 추가 DB_1',
    r'G:\경구약제 DB\0.단일 경구약제 1000종\경구약제 1000종 추가 DB_2',
]

def copy_crop_pill_from_json(args, pathfile_src_json, pathfile_src_png, pathfile_dest_json, pathfile_dest_png):
    if not pathfile_src_png.exists() :
        args.logger.info(f"file does'nt exist : {str(pathfile_src_png)}")
        return

    # json copy
    shutil.copyfile(str(pathfile_src_json), str(pathfile_dest_json))

    # crop
    dict_pill_info = utils.read_dict_from_json(str(pathfile_src_json))
    bbox = dict_pill_info['annotations'][0]['bbox']
    np_bbox = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]])
    np_center = np.average(np_bbox, axis=0)
    diff =  np.max(np_bbox[1] - np_bbox[0])
    np_low = np_center - diff/2
    np_high = np_center + diff/2
    np_high = np_high.astype(np.int32)
    np_low = np_low.astype(np.int32)

    image_cv = utils.open_opencv_file(str(pathfile_src_png))
    image_cv_cropped = image_cv[np_low[1]:np_high[1], np_low[0]:np_high[0]]
    image_cv_resized = cv2.resize(image_cv_cropped, (args.size_image, args.size_image))

    utils.save_opencv_file(image_cv_resized, str(pathfile_dest_png))

def copy_crop_pill_from_org(args):
    '''
    list_dir_pill_org 의 directory은  알약  dir들을 여러 개 포함하고,
    알약 dir에는   조건에 따라, 1200여개의 png가 있다.
    이를  아래 pathdir_dest dir에 모든다.
    단 이미지를 bbox기준으로, 224x224으로 crop한다.
    :param args:
    :return:
    '''
    pathdir_dest = Path(r'G:\proj_pill_data\pill_data_croped')

    for dir_pill_org in list_dir_pill_org :
        list_count = [ 1 for _ in Path(dir_pill_org).iterdir()]
        count_dir = len(list_count)
        index_dir = 0
        for pathdir_pill_class_json in Path(dir_pill_org).iterdir():
            print(f'working dir index is { index_dir}/{count_dir}, {str(pathdir_pill_class_json)}')
            index_dir += 1
            if not 'json' in pathdir_pill_class_json.name or not pathdir_pill_class_json.is_dir():
                continue

            no_json = pathdir_pill_class_json.name.split('_')[0]
            pathdir_pill_class = pathdir_pill_class_json.with_name(no_json)
            pathdir_dest_class = pathdir_dest.joinpath(no_json)
            pathdir_dest_class.mkdir(exist_ok=True)

            for i,  pathfile_pill_json in enumerate(pathdir_pill_class_json.glob('*.json')):
                # if i == 0 :
                #     shutil.copy(pathfile_pill_json,pathdir_dest )
                pathfile_pill_png = pathdir_pill_class.joinpath( pathfile_pill_json.stem + '.png')
                copy_crop_pill_from_json(args, pathfile_pill_json, pathfile_pill_png, pathdir_dest_class.joinpath(pathfile_pill_json.stem + '.json'), pathdir_dest_class.joinpath(pathfile_pill_json.stem + '.png'))

if __name__ == '__main__':
    # job = 'hrnet_w64'
    job = 'resnet152'
    args = get_cli_args(job=job, run_phase='train', aug_level=1 )
    args.logger = utils.create_logging(args.file_log)
    copy_crop_pill_from_org(args)
    print('done')