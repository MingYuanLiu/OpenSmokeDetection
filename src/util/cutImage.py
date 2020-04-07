import cv2
import os
import argparse
import random



def main(args):
    src_img = args.src
    save_dir = args.save_dir
    print("src img is {} and save dir is {}".format(src_img, save_dir))
    img = cv2.imread(src_img)
    w = img.shape[0]
    h = img.shape[1]
    stride = 100
    num = random.randint(0,100)
    assert(w > stride and h > stride)
    for x in range(0, w-stride, stride):
        for y in range(0, h-stride, stride):
            img_name = "cut_img_" + str(num) + str(x) + "_" + str(y)
            small_img = img[x:x+stride, y:y+stride, :]
            cv2.imwrite(os.path.join(save_dir, img_name) + ".jpg",small_img)
            print("save img to -", img_name)
    
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut Image into small block")
    parser.add_argument("-src", default="", metavar="FILE", help="path to source image")
    parser.add_argument("--save-dir", default="", help="dir to save cut image")
    args = parser.parse_args()
    main(args)

    