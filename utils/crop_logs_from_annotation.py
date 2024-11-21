import cv2
import json 
import os 
import uuid

def load_json(json_path: str):
    with open(json_path, 'r') as read_json:
        data = json.load(read_json)
    return data

def get_file_by_extension(target_folder: str, extensions: tuple):
    '''
    Takes in a target folder and a tuple of extensions,
    returns a list of path strings of files with the specified extensions,
    including those in subfolders.
    '''
    _a = []
    for root, _, files in os.walk(target_folder):
        for file in files:
            if file.endswith(extensions):
                _a.append(os.path.join(root, file))
    return _a

def find_and_crop_label(target_folder: str, dest_folder: str, label_name: tuple):
    all_file = get_file_by_extension(target_folder, (".json"))
    for file in all_file:
        data = load_json(file)
        try:
            _load_image = cv2.imread(os.path.join(target_folder, data["imagePath"]))
            for shape in data["shapes"]:
                if shape["label"] == label_name:
                    x1 , y1 = shape["points"][0]
                    x2 , y2  = shape["points"][1]
                    cropped_image = _load_image[int(y1):int(y2), int(x1):int(x2)]
                    cv2.imwrite(os.path.join(dest_folder, f"{uuid.uuid4().hex}.png"), cropped_image)
        except Exception as e:
            continue

if __name__ == "__main__":
    # jsons = find_and_crop_label("/home/hbdesk/pubg_parser/stream_saver/outputs/filtered_logs3", "/home/hbdesk/pubg_parser/stream_saver/outputs/cropped_logs3", ("LOG"))
    # jsons = find_and_crop_label("/home/hbdesk/pubg_640_dataset/test_logs", "/home/hbdesk/pubg_640_dataset/test_names", ("name"))
    # jsons = find_and_crop_label("/media/hbdesk/UNTITLED/test_sets/640_export3/", "/media/hbdesk/UNTITLED/test_sets/logs4", ("LOG"))
    jsons = find_and_crop_label("/media/hbdesk/UNTITLED/test_sets/logs4", "/media/hbdesk/UNTITLED/test_sets/names4", ("name"))
    print(jsons)
