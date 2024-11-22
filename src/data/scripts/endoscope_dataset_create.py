import os
import cv2
import numpy as np


def save_sequence(path, save_path):
    cap = cv2.VideoCapture(path)

    i = 0
    success = True
    if cap.isOpened():
        while(success):
            print(i)
            success, frame = cap.read()
            if i % 360 < 120:
                # print(save_path + str(i // 360) + '/' + str(i % 360).zfill(4) + '.jpg')
                if not os.path.exists(save_path + str(i // 360)):
                    os.makedirs(save_path + str(i // 360))
                cv2.imwrite(save_path + str(i // 360) + '/' + str(i % 360).zfill(4) + '.jpg', frame)
            i += 1
    cap.release()

def show_sequence(path):
    imgs = []
    for i in range(480):
        print(os.path.join(path, str(i), '0060.jpg'))
        img = cv2.imread(os.path.join(path, str(i), '0060.jpg'))
        print(img.shape)
        cv2.imshow(f'{i}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        img.resize((48, 48, 3))
        imgs.append(img)
    imgs = np.stack(imgs)
    print(imgs.shape)

    for i in range(8):
        column = []
        for j in range(6):
            row = []
            for k in range(10):
                print(imgs[60*i + 10*j + k].shape)
                row.append(imgs[60*i + 10*j + k])
            row = np.hstack(row)
            column.append(row)
        column = np.vstack(column)
        print(column.shape)
        cv2.imshow(f'{i}', column)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    path = 'E:/PELD/enhancing/data/Endoscope/origin/real/Encode_720P_20_Tot.mp4'
    save_path = 'E:/PELD/enhancing/data/Endoscope/frame2/'
    test_path = 'E:/PELD/enhancing/data/Endoscope/frame/'

    # save_sequence(path, save_path)

    show_sequence(test_path)
