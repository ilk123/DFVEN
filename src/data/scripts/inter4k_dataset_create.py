import os
import cv2
import shutil
import numpy as np


def save_sequence(path, save_path):
    num = 0
    for n in range(1000):
        cap = cv2.VideoCapture(os.path.join(path, f'{n+1}.mp4'))

        i = 0
        success = True
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                if (i % 20) % 4 == 0 and (i % 20) // 4 < 4:
                    print(os.path.join(save_path, f'{num}', str(i % 20).zfill(4) + '.jpg'))
                    if not os.path.exists(save_path + f'{num}'):
                        os.makedirs(save_path + f'{num}')
                    cv2.imwrite(os.path.join(save_path, f'{num}', str(i % 20).zfill(4) + '.jpg'), frame)
                if i % 20 == 19:
                    num += 1
                i += 1
        cap.release()

def combine_sequence(path, res_path):
    num = 15
    t = 0
    for i in range(14841):
        if i % (num * 10) == 0:
            t += 1
            os.makedirs(os.path.join(res_path, f'{t}'))
        if (i % num) % 5 == 0:
            for j in [0, 4, 8, 12]:
                print(os.path.join(path, f'{i}', f'{j:04d}.jpg'))
                print('\r', os.path.join(res_path, f'{t}', f'{(j//4+i//5*4)%120:04d}.jpg'))
                shutil.copyfile(os.path.join(path, f'{i}', f'{j:04d}.jpg'), os.path.join(res_path, f'{t}', f'{(j//4+i//5*4)%120:04d}.jpg'))



if __name__ == '__main__':

    path = 'E:/PELD/enhancing/data/Inter4K/60FPS_UHD'
    save_path = 'E:/PELD/enhancing/data/Inter4K/frame/'
    res_path = 'E:/PELD/enhancing/data/Inter4K/frame2/'

    save_sequence(path, save_path)
    combine_sequence(save_path, res_path)
