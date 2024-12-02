import os
import csv

def generate_file_paths_and_labels(root_dir):
    """
    遍历根目录中的文件夹，生成文件路径和对应的数字标签。

    参数:
        root_dir (str): 数据集根目录路径。

    返回:
        file_paths (list): 所有音频文件的路径列表。
        labels (list): 对应的数字标签列表。
        label_mapping (dict): 文件夹名称到数字标签的映射。
    """
    file_paths = []
    labels = []
    label_mapping = {}
    current_label = 0

    # 遍历根目录下的每个文件夹
    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 跳过非文件夹

        # 为每个类别分配一个数字标签
        label_mapping[folder_name] = current_label

        # 遍历当前类别文件夹中的所有音频文件
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.wav', '.mp3', '.flac')):  # 支持的音频格式
                file_paths.append(os.path.join(folder_path, file_name))
                labels.append(current_label)

        current_label += 1

    return file_paths, labels, label_mapping


def save_labels_to_csv(file_paths, labels, output_path):
    """
    将文件路径和标签保存到 CSV 文件。

    参数:
        file_paths (list): 文件路径列表。
        labels (list): 对应的数字标签列表。
        output_path (str): 输出 CSV 文件路径。
    """
    with open(output_path, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        # 写入表头
        writer.writerow(['file_path', 'label'])
        # 写入每行数据
        writer.writerows(zip(file_paths, labels))
    print(f"标签数据已保存到 {output_path}")


if __name__ == "__main__":
    # 设置数据集根目录
    root_dir = "PFCIC_Dataset_Sliced"

    # 生成文件路径和标签
    file_paths, labels, label_mapping = generate_file_paths_and_labels(root_dir)

    # 打印生成的标签映射
    print("类别到标签的映射:", label_mapping)
    print("文件路径示例:", file_paths[:5])
    print("标签示例:", labels[:5])

    # 保存标签数据到 CSV 文件
    output_csv_path = "audio_labels.csv"
    save_labels_to_csv(file_paths, labels, output_csv_path)
