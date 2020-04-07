import os
import argparse


def main(args):
    img_dir = args.dir
    annotationFile = args.annotation
    file_handler = open(annotationFile, mode='w')

    for class_name in os.listdir(img_dir):
        if class_name[0] == '.':
            continue
        img_subdir = os.path.join(img_dir, class_name)
        if class_name == "smoke":
            label = 1
        else:
            label = 0
        for img in os.listdir(img_subdir):
            img_path = os.path.join(img_subdir, img)
            tag = str(label) + " " + img_path + "\n"
            file_handler.write(tag)

    print("DONE")
    file_handler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", help="the dir of dataset")
    parser.add_argument("--annotation", help="the txt file of annotation")
    args = parser.parse_args()
    main(args)




