import os

base_dir = os.path.abspath(os.path.dirname(__file__))


def compress_videos(dataset_path, output_path):
    for i in os.listdir(data_path):
        class_path = os.path.join(data_path, i)
        out_class_path = os.path.join(output_path, i)
        if not os.path.exists(out_class_path):
            os.makedirs(out_class_path)
        for j in os.listdir(class_path):
            video_path = os.path.join(class_path, j)
            out_video_path = os.path.join(out_class_path, j)
            cmd = f"ffmpeg -i {video_path} -c:v libx264 -preset veryfast -vf scale=720:1280 {out_video_path}"
            print(cmd)
            os.system(cmd)

if __name__ == "__main__":
    data_path = r"D:\Desktop\server\CSL-1000"
    output_path = os.path.join(base_dir,"../../CSL-1000")
    compress_videos(dataset_path=data_path, output_path=output_path)