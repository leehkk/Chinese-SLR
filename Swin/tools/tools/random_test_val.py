import os
import csv
import random


def get_train_val_test(datasets_dir,labels_dir, csv_path, class_num):
    with open(os.path.join(labels_dir, "train.csv"), 'w', newline='') as train:
        with open(os.path.join(labels_dir, "val.csv"), 'w', newline='') as val:
            with open(os.path.join(labels_dir, "test.csv"), 'w', newline='') as test:
                train_writer = csv.writer(train)
                val_writer = csv.writer(val)
                test_writer = csv.writer(test)
                count = 0
                for class_name in os.listdir(datasets_dir):
                    if count >= class_num:
                        break
                    with open(csv_path, "r", newline="") as csv_file:# 读class
                        reader = csv.reader(csv_file)
                        rows = [row for row in reader]
                        for match in rows:
                            if class_name == match[0]:
                                class_idx = match[1]
                    class_path = os.path.join(datasets_dir,class_name)
                    files = os.listdir(class_path)
                    #
                    # for j in os.listdir(class_path):
                    #     video_path = os.path.join(class_path,j)
                    #     print(video_path)
                    #     content = [video_path, class_idx]
                    #     if j.split("_")[-1] == "8.MOV" or j.split("_")[-1] == "8.mov":
                    #         test_writer.writerow(content)
                    #     elif j.split("_")[-1] == "9.mp4":
                    #         val_writer.writerow(content)
                    #     else:
                    #         train_writer.writerow(content)
                    random_list = random.sample(range(len(files)), len(files))
                    for i in range(len(random_list)):
                        path = files[random_list[i]]
                        video_path = os.path.join(class_path,path)
                        content = [video_path,class_idx]
                        print(content)
                        if i < 8:
                            train_writer.writerow(content)
                        elif i < 9:
                            val_writer.writerow(content)
                        else:
                            test_writer.writerow(content)
                    count += 1

if __name__ == "__main__":
    class_num = 100
    datasets_dir = "F:/uestc/code/dataset/CSL-1000" #数据集位置
    #flow_dir = "F:/uestc/code/dataset/CSL-flow"    #
    class_dir = "F:/uestc/dataset/leshan/class.csv"    #类别位置
    labels_dir = os.path.join("C:/uestc/code/TimeSformer-main", f"new_datasets/CSL-{class_num}")  #保存位置
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    get_train_val_test(datasets_dir, labels_dir, class_dir, class_num)
