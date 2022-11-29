from dataset_generator import DatasetGenerator, Camera
import os
from termcolor import cprint

if __name__ == '__main__':


    checkpoint_path = 'model_weights.ckpt'
    output_folder = "output"
    dataset_path = "/home/alessio/dev/DLO-WSL/data_test"
    imgs_folder = "images"
    tracepen_folder = "points"

    gen = DatasetGenerator(  checkpoint_path=checkpoint_path, 
                            dataset_main_path=dataset_path, 
                            imgs_folder=imgs_folder,
                            tracepen_folder=tracepen_folder,
                            output_folder=output_folder,
                            camera_model=Camera)  
    

    gen.diameter_path = os.path.join(gen.dataset_path, "diameter.txt")
    with open(gen.diameter_path) as f:
        gen.diameter_data = f.readlines()

    for e in gen.diameter_data:
        if e == "\n":
            break

        x = e.split(" ")
        cable_type = x[0]

        cable_diam = float(x[-1].split("\n")[0]) / 1000
        gen.diameter = cable_diam

        cprint("cable type: {}, diameter: {} m".format(cable_type, cable_diam), "yellow")

        gen.load_pen_raw_points_dataset(cable=cable_type)
        cprint("pen raw points loaded!", "yellow")

        gen.one_step_run(cable=cable_type)
