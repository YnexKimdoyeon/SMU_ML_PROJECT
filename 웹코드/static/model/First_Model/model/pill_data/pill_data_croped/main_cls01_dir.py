from pill_classifier import *
from get_cli_args import get_cli_args
from pathlib import Path
from PIL import Image
import os

class Dataset_Dir(Dataset):
    def __init__(self, args, dir_dataset, transform=None, target_transform=None, run_phase='train'):
        self.args = args
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.target_transform = target_transform

        self.list_images = [ png.name  for png in Path(dir_dataset).iterdir() if png.suffix == '.png']
        self.run_phase = run_phase

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir_dataset, self.list_images[idx])).convert('RGB')  # <-- 중요
        label = 0
        path_img = self.list_images[idx]
        aug_name = ""

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.run_phase == 'valid' or self.run_phase == 'test':
            return image, label, path_img, aug_name
        else:
            return image, label

#됬는데 모델 삑남 ㅅㅂ
#이새기 못맞췄는데? ?ㅈㅈ 이거 처음부터 잘못된 거 같은데
# ㄴㄴ 잘못된건없는데 그냥 ㅈ같다 ㅅㅂ 개추 개추 아 죽고싶다 ㅇㅈㄱㅅㄷ ㅇㅈㅅㄱㄷ
# 아씨발 못맞추는데요
#아니 이거 모델만든사람이 ㅈ같이 만들었어
#배경색 존나 많이들어가게 이렇게 라벨링을 해두면 모델이 배경까지 학습해서 못맞추지 씻팔
#일단 오늘은 여기까지
job = 'resnet152'
if __name__ == '__main__':
    # job = 'hrnet_w64'
    job = 'resnet152'
    args = get_cli_args(job=job, run_phase='test', aug_level=0, dataclass='01')

    print(f'model_path_in is {args.model_path_in}')

    dir_testimage = r'dir_testimage'

    args.dataset_valid = Dataset_Dir(args, dir_testimage, transform=transform_normalize, run_phase='test' if args.run_phase == 'test' else 'valid')
    args.batch_size = len(args.dataset_valid)
    args.verbose = False
    print(f'valid dataset was loaded')

    pill_classifier(args)
    #뭐지?
    #이거만 찾아볼래?
    #이미지 학습할때 사용한 이미지만 에러안남
    #아마 이미지 포맷문제인듯 이미지 가로세로 픽셀크기나 뭐 그런거
    #찾아봐주셈 아까 봤을때는 244 244 아니면 라스넷이 못먹는다 햇던거같은데 224 인가 ㅋㅋ 저기있는 이미지랑 똑같은 크기로 맞춘건데 ㄷㄷ ㄱㄷ
    #ㄱㄷ 카톡열어줘봐
    print(args.list_preds)
    print('job done')