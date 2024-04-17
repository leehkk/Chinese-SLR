import csv
import os


def slovo_get_dataset(csv_path, save_path):
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        with open(os.path.join(save_path,"train.csv"), "w", newline="") as train_file:
            train_writer = csv.writer(train_file, lineterminator='\n')
            with open(os.path.join(save_path, "test.csv"), "w", newline="") as test_file:
                test_writer = csv.writer(test_file, lineterminator='\n')
                class_name_list = []
                class_label_list = []
                label = 0
                for index, row in enumerate(reader):
                    if index == 0:
                        continue
                    row_list = row[0].split("\t")
                    if len(row_list) < 7:
                        row_list2 = row[1].split("\t")
                        row_list[1] = row_list[1] + row_list2[0]
                        row_list.extend(row_list2[1:])
                    print(row_list)
                    file_name, class_name, frames, train = row_list[0], row_list[1], row_list[5], row_list[6]
                    if class_name == "no_event":
                        continue
                    if class_name not in class_name_list:
                        class_name_list.append(class_name)
                        class_label_list.append(label)
                        if train == "True":
                            train_writer.writerow([file_name,label])
                        elif train == "False":
                            test_writer.writerow([file_name,label])
                        label += 1
                    else:
                        cur_label = class_name_list.index(class_name)
                        if train == "True":
                            train_writer.writerow([file_name,cur_label])
                        elif train == "False":
                            test_writer.writerow([file_name,cur_label])
                with open(os.path.join(save_path, "class.csv"), "w", newline="") as class_file:
                    class_writer = csv.writer(class_file, lineterminator='\n')
                    for i in range(len(class_name_list)):
                        class_writer.writerow([class_name_list[i],class_label_list[i]])



if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    csv_path = os.path.join(base_dir,"../../../slovo/annotations.csv")
    save_path = base_dir
    slovo_get_dataset(csv_path,save_path)
