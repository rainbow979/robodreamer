import os
import glob

def get_paths(root="../berkeley"):
    f = []
    for dirpath, dirname, filename in os.walk(root):
        if "image" in dirpath:
            f.append(dirpath)
    print(f"Found {len(f)} sequences")
    return f

def get_paths_from_dir(dir_path):
    paths = glob.glob(os.path.join(dir_path, 'im*.jpg'))
    try:
        paths = sorted(paths, key=lambda x: int((x.split('/')[-1].split('.')[0])[3:]))
    except:
        print(paths)
    return paths

def calc_parameters(net):
    total_params = 0
    for param in net.parameters():
        total_params += param.numel()
    return total_params


def cycle(dl):
    while True:
        for data in dl:
            yield data

