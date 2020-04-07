import cv2
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def getDataShape(args):
    """
    统计样本集的形状分布
    """
    img_dir = args.dir
    assert (os.path.isdir(img_dir))
    shapes = []
    num_bin_list = [0,0,0,0]
    for img in os.listdir(img_dir):
        if img[-4:] != '.jpg' and img[-4:] != '.png' and img[-5:] != '.jpeg':
            print("Do not support this type.")
            continue
        img_path = os.path.join(img_dir, img)
        if not os.path.exists(img_path):
            print("Not Found this File")
            continue
        imgMat = cv2.imread(img_path)
        w, h, _ = imgMat.shape
        if w not in shapes:
            shapes.append(w)
        
        if w <= 25 and h <= 25:
            num_bin_list[0] += 1
        if w >= 25 and w < 50 and h >= 25 and h < 50:
            num_bin_list[1] += 1
        if w >= 50 and w < 100 and h >= 50 and h < 100:
            num_bin_list[2] += 1
        if w >= 100 and h >= 100:
            num_bin_list[3] += 1
    
    print("shapes is {}".format(shapes))
    print("num_list is {}".format(num_bin_list))
    list_sum = sum(num_bin_list)
    num_bin_list = [100 * c / list_sum for c in num_bin_list]
    rects = plt.bar(len(num_bin_list), num_bin_list, color='rgby')
    name_list = ['25x25', '50x50','100x100','more']
    index = [float(c) for c in range(len(num_bin_list))]
    plt.ylim(ymax=100, ymin=0)
    plt.ylabel('numbers(%)')
    plt.xticks(index, name_list)

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height) + "%", ha='center', va='bottom')
    plt.show()

def clipImageToFixedSize(args):
    """
    将图片的大小钳制在50x50
    """
    img_dir = args.dir
    dst_save_dir = args.save_dir
    if not os.path.exists(dst_save_dir):
        os.mkdir(dst_save_dir)
    
    assert (os.path.isdir(img_dir))
    for img_name in os.listdir(img_dir):
        if img_name[-4:] != '.jpg' and img_name[-4:] != '.png' and img_name[-5:] != '.jpeg':
            print("Do not support this type.")
            continue
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print("Not Found this File")
            continue
        img = cv2.imread(img_path)
        w, h = img.shape[:2]
        if np.round(w/10) == 5 or np.round(h/10) == 50:
            img = cv2.resize(img, (50, 50))
            save_path = os.path.join(dst_save_dir, img_name)
            cv2.imwrite(save_path, img)
        
        if np.round(w/10) == 10 or np.round(h/10) == 10:
            img = cv2.resize(img, (100, 100))
            cut_imgs = np.array([img[x:x + 50, y:y + 50,:] for x in [0, 50] for y in [0, 50]])
            for i, cut_img in enumerate(cut_imgs):
                save_img_name = img_name[:-4] + str(i) + '.jpg'
                save_img_path = os.path.join(dst_save_dir, save_img_name)
                cv2.imwrite(save_img_path, cut_img)
            
    print('DONE!')
            
            
            






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", dest='dir', help='dir of image')
    parser.add_argument("--save-dir", dest='save_dir', help='dir of dst image')
    args = parser.parse_args()
    clipImageToFixedSize(args)

        

    
