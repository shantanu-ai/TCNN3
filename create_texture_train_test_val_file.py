import os
import shutil


def do_create_files(root_path, dst_dir_root, label_dir):
    for i in range(10):
        # get the file names from labels folder
        file_name = root_path + label_dir + str(i + 1) + ".txt"
        with open(file_name) as f:
            for source_file in f.readlines():
                folder_name = source_file.split("/")[0]
                dst_root = dst_dir_root + str(i + 1) + "/"
                dst_dir = dst_dir_root + str(i + 1) + "/" + folder_name + "/"

                if not os.path.isdir(dst_root) and not os.path.isdir(dst_dir):
                    os.mkdir(dst_root)

                if not os.path.isdir(dst_dir):
                    os.mkdir(dst_dir)

                source_file_full_name = root_path + "images/" + source_file.rstrip("\n")
                print(dst_dir)
                print(source_file_full_name)

                # copy image
                shutil.copy(source_file_full_name, dst_dir)


if __name__ == '__main__':
    ROOT_PATH = "./images"

    # test path
    test_dst_dir_root = ROOT_PATH + "test"
    test_label_dir = "labels/test"

    # val path
    val_dst_dir_root = ROOT_PATH + "val"
    val_label_dir = "labels/val"

    # train path
    train_dst_dir_root = ROOT_PATH + "train"
    train_label_dir = "labels/train"

    # create folders
    do_create_files(ROOT_PATH, test_dst_dir_root, test_label_dir)
    do_create_files(ROOT_PATH, train_dst_dir_root, train_label_dir)
    do_create_files(ROOT_PATH, val_dst_dir_root, val_label_dir)
