import numpy as np
import cv2, glob, os, torch, pickle
import matplotlib.pyplot as plt
import utils
from models import *
from termcolor import cprint
from tqdm import tqdm

class Corrector():

    def __init__(self, checkpoint_path, crop_size=96):
        self.crop_size = crop_size

        # Load Model
        self.network = ResNetGaussian(output_dim=crop_size)
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        self.network.load_state_dict(state_dict)
        self.network.eval()
       
    
    def predict(self, image_crop):
        img_tensor = torch.tensor(image_crop / 255.).permute(2, 0, 1).float().unsqueeze(0)
        return self.network(img_tensor).sigmoid().squeeze().detach().cpu().numpy()


    def process_crop(self, img_crop, angle, TH=0.2, debug=True):
        pred = self.predict(img_crop)
        apred = np.argmax(pred)

        x_left = None
        for i in range(apred, self.crop_size):
            if pred[i] < TH:
                x_left = i
                break

        x_right = None
        for i in range(apred, 0, -1):
            if pred[i] < TH:
                x_right = i
                break

        if x_left is None and x_right is not None:
            pred_offset = - (apred - self.crop_size//2)
        elif x_right is None and x_left is not None:
            pred_offset = - (apred - self.crop_size//2)
        else:
            pred_offset = - ((x_left+x_right)/2 - self.crop_size//2)


        offset_x = np.cos(angle) * pred_offset
        offset_y = np.sin(angle) * pred_offset

        if debug:
            print("left: {}, right: {}, max value: {}".format(x_left, x_right, np.max(pred)))

            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            ax1.scatter(self.crop_size//2,self.crop_size//2, label="center", s=80)
            ax1.scatter(self.crop_size//2 - pred_offset, self.crop_size//2, label="corrected", s=80)
            ax1.legend()
            ax2.scatter([i for i in range(self.crop_size)], pred)
            plt.tight_layout(pad=0.1)
            plt.show()

        return offset_x, offset_y, np.max(pred)


    def exec(self, points_2d_dict, img, TH=0.2, debug=False):

        ext = self.crop_size
        img_ext = cv2.copyMakeBorder(img, top=ext, bottom =ext, left=ext, right=ext, borderType = cv2.BORDER_REFLECT_101)

        # CORRECTION WITH NEURAL NETWORK
        out_dict = {k: None for k,_ in points_2d_dict.items()}
        for it, point in points_2d_dict.items():

            crop_out = self.get_crop(img_ext.copy(), point, points_2d_dict, key=it)     
            if crop_out is not None:
                img_crop, angle = crop_out
                offset_x, offset_y, _ = self.process_crop(img_crop, angle, TH=TH, debug=debug)
                out_dict[it] = tuple([int(point[0]-offset_x), int(point[1]-offset_y)])
                
        points_2d_corrected_dict = {k: v for k,v in out_dict.items() if v is not None}
        return points_2d_corrected_dict

    def exec_fixed_dir(self, p_dict, dir, img, max_th=0.5, debug=False):

        ext = self.crop_size
        img_ext = cv2.copyMakeBorder(img, top=ext, bottom =ext, left=ext, right=ext, borderType = cv2.BORDER_REFLECT_101)

        # CORRECTION WITH NEURAL NETWORK
        out_dict = {k: None for k,_ in p_dict.items()}
        for it, point in p_dict.items():

            crop_out = self.get_crop_fixed_dir(img_ext.copy(), point, dir)       
            if crop_out is not None:
                img_crop, angle = crop_out
                offset_x, offset_y, max_score = self.process_crop(img_crop, angle, debug=debug)
                if max_score < max_th:
                    break

                out_dict[it] = tuple([int(point[0]-offset_x), int(point[1]-offset_y)])
                
        points_2d_corrected_dict = {k: v for k,v in out_dict.items() if v is not None}
        return points_2d_corrected_dict



    def get_crop(self, img_ext, point, points_2d_dict, key):
        if point is None:
            return None

        if points_2d_dict.get(key-1) is not None:
            p2 = points_2d_dict.get(key-1)
        elif points_2d_dict.get(key+1) is not None:
            p2 = points_2d_dict.get(key+1)
        else:
            print("error!")
            return None
 
        p = np.array(point)
        p2 = np.array(p2)

        if p[0] == p2[0] and p[1] == p2[1]:
            return None

        CS = self.crop_size
        CS_H = self.crop_size // 2

        # account for border wrap
        p[0] += CS
        p[1] += CS
        p2[0] += CS
        p2[1] += CS

        dir = p2 - p
        dir = dir / np.linalg.norm(dir)
        angle = np.arctan2(dir[1], dir[0]) - np.pi/2  
        angle = np.degrees(angle)

        crop_img_big = img_ext[p[1]-CS:p[1]+CS, p[0]-CS:p[0]+CS]
        if crop_img_big.shape != (CS*2, CS*2, 3):
            return None

        img_rotated = utils.rotate_image(crop_img_big.copy(), angle)
        crop_img_rotated = img_rotated[CS-CS_H:CS+CS_H, CS-CS_H:CS+CS_H]
        if crop_img_rotated.shape != (CS, CS, 3):
            return None
        
        return crop_img_rotated, np.deg2rad(angle)



    def get_crop_fixed_dir(self, img_ext, point, dir):
        if point is None:
            return None
        
        p = point.copy()

        CS = self.crop_size
        CS_H = self.crop_size // 2

        p[0] += CS
        p[1] += CS

        dir = dir / np.linalg.norm(dir)
        angle = np.arctan2(dir[1], dir[0]) - np.pi/2  
        angle = np.degrees(angle)

        crop_img_big = img_ext[p[1]-CS:p[1]+CS, p[0]-CS:p[0]+CS]
        if crop_img_big.shape != (CS*2, CS*2, 3):
            return None

        img_rotated = utils.rotate_image(crop_img_big.copy(), angle)
        crop_img_rotated = img_rotated[CS-CS_H:CS+CS_H, CS-CS_H:CS+CS_H]
        if crop_img_rotated.shape != (CS, CS, 3):
            return None
        
        return crop_img_rotated, np.deg2rad(angle)



class EdgeCorr():

    def __init__(self):
        self.CS = 15

    def exec(self, points, img):
        points_s, points_sd = utils.compute_spline_2(points)
        N = np.linalg.norm(points_sd, axis=1)

        new_points = []
        for it, point in enumerate(points_s):
            dir = points_sd[it]
            dir[0] = dir[0] / N[it]
            dir[1] = dir[1] / N[it]

            crop_out, angle_out = self.get_crop(point, dir, img)
            grad_y = cv2.Sobel(cv2.cvtColor(crop_out, cv2.COLOR_BGR2GRAY), cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            vec = abs_grad_y[:,self.CS//2]

            offset = np.argmax(vec) - self.CS//2
            offset_x = np.sin(angle_out) * offset
            offset_y = np.cos(angle_out) * offset

    
            new_points.append(tuple([point[0]-offset_x, point[1]+offset_y]))
        return np.array(new_points)


    def get_crop(self, point, dir, img):          
        angle = np.arctan2(dir[1], dir[0])
        point = point.astype(int)
        crop_img_big = img[point[1]-self.CS:point[1]+self.CS, point[0]-self.CS:point[0]+self.CS]
        crop_img_rotated = utils.rotate_image(crop_img_big.copy(), np.degrees(angle))
        crop_img_rotated = crop_img_rotated[self.CS-self.CS//2:self.CS+self.CS//2, self.CS-self.CS//2:self.CS+self.CS//2]
        return crop_img_rotated, angle




class Camera():
    K = np.array([856.657396, 0.0 , 611.745622, 0.0, 858.802578, 514.072871, 0.0, 0.0, 1.0]).reshape(3,3)
    W = 1280
    H = 1024    
    D = np.array([-0.248306,  0.124092,  0.000316, -0.000393, -0.038915])


class DatasetGenerator():

    def __init__(self, checkpoint_path, dataset_main_path, tracepen_folder="pen", imgs_folder="green", output_folder="out", camera_model=Camera):
        self.camera_model = camera_model
        self.dataset_path = dataset_main_path
        self.tracepen_path = os.path.join(self.dataset_path, tracepen_folder)
        self.imgs_path = os.path.join(self.dataset_path, imgs_folder)
        self.output_path = os.path.join(self.dataset_path, imgs_folder + "_" + output_folder)
        os.makedirs(self.output_path, exist_ok=True)

        self.pen_points = []
        self.corrector_net = Corrector(checkpoint_path=checkpoint_path)
        self.edge_corr = EdgeCorr()
        self.diameter = 8/1000


    def load_pen_raw_points_user(self):
        pen_files = sorted([int(f.split("/")[-1].split("_")[0]) for f in glob.glob(os.path.join(self.tracepen_path, "*"))])
        self.pen_points = [np.loadtxt(os.path.join(self.tracepen_path, str(f).zfill(4) + "_point.txt")) for f in pen_files]
        self.pen_points = np.array([[p[0], p[1], p[2]-self.diameter] for p in self.pen_points])


    def load_pen_raw_points_dataset(self, cable=None):
        pen_files = sorted([int(f.split("/")[-1].split("_")[0]) for f in glob.glob(os.path.join(os.path.join(self.tracepen_path, cable), "*"))])
        pen_points = [np.loadtxt(os.path.join(os.path.join(self.tracepen_path, cable), str(f).zfill(4) + "_point.txt")) for f in pen_files]

        self.pen_points = np.array([[p[0], p[1], p[2]-self.diameter] for p in pen_points])
        self.pen_points_dict = {it: [] for it, p in enumerate(self.pen_points)}


    def extension_endpoints(self, points_2d_dict):

        def get_points_line(point_start, line_dir, length=50, step=1):
            ls = np.arange(0, length, step).reshape(-1, 1)
            ls = np.repeat(ls, 2, axis=1)
            D = np.repeat(line_dir.reshape(1,-1), ls.shape[0], axis=0)
            return (point_start + D * ls).astype(int)


        sub_points = list(points_2d_dict.values())
        start_point = np.array(sub_points[0])
        start_nn_point = np.array(sub_points[1])
        tail_point = np.array(sub_points[-1])
        tail_nn_point = np.array(sub_points[-2])

        # begin
        dir_begin = start_point - start_nn_point
        dir_begin = dir_begin / np.linalg.norm(dir_begin)
        points_begin = get_points_line(start_point, dir_begin)

        # tail
        dir_end = tail_point - tail_nn_point
        dir_end = dir_end / np.linalg.norm(dir_end)
        points_tail = get_points_line(tail_point, dir_end)
        
        return points_begin, -dir_begin, points_tail, -dir_end




    def process_image_end(self, id, pen_points, cable=None, use_distortion=True, enable_extension=True):

        # LOAD IMG AND POSE
        img_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_image.png")
        pose_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_pose.txt")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pose = np.loadtxt(pose_path)

        # PROJECTION
        points_2d_input = utils.projection(pose, pen_points, self.camera_model, use_distortion=use_distortion, keep_invalid=True)

        # CORRECTION WITH NEURAL NETWORK
        points_2d_dict = {it: points_2d_input[it] for it, p in enumerate(pen_points) if points_2d_input[it] is not None}
        points_2d_corrected_dict = self.corrector_net.exec(points_2d_dict, img=img)

        #######################################
        # EXTENSION AT ENDPOINTS
        if enable_extension:
            points_begin, dir_begin, points_tail, dir_tail = self.extension_endpoints(points_2d_corrected_dict)
            test_dict_begin = {it: p for it, p in enumerate(points_begin)}
            test_dict_begin_corr = self.corrector_net.exec_fixed_dir(test_dict_begin, dir_begin, img=img, debug=False)

            test_dict_tail = {it: p for it, p in enumerate(points_tail)}
            test_dict_tail_corr = self.corrector_net.exec_fixed_dir(test_dict_tail, dir_tail, img=img, debug=False)

            new_head, new_tail = None, None
            if test_dict_begin_corr:
                new_head = list(test_dict_begin_corr.values())[-1]
            if test_dict_tail_corr:
                new_tail = list(test_dict_tail_corr.values())[-1]

            final_dict = {}
            if new_head is not None:
                final_dict[0] = {"point_pen": pen_points[0], "point_2d": new_head}

            for it, p in enumerate(pen_points):
                point_2d = points_2d_corrected_dict.get(it)
                if point_2d is not None:
                    final_dict[it+1] = {"point_pen": p, "point_2d": point_2d}
            
            if new_tail is not None:
                final_dict[len(pen_points)+1] = {"point_pen": pen_points[-1], "point_2d": new_tail}

        else:
            final_dict = {}
            for it, p in enumerate(pen_points):
                point_2d = points_2d_corrected_dict.get(it)
                if point_2d is not None:
                    final_dict[it+1] = {"point_pen": p, "point_2d": point_2d}
        

        ###########################################################################################
        # PROJECT CORRECTED POINTS IN 3D FOR COMPUTING DLO BORDERS
        points_3d_corr = []
        for k,v in final_dict.items():
            t_w = np.append(v["point_pen"], 1)
            t_cam = (np.matmul(np.linalg.inv(pose), t_w))[:3]

            points_out = utils.unproject(np.array([v["point_2d"]]), np.array([t_cam[2]]), self.camera_model, use_distortion=use_distortion)

            t_point_cam = np.append(points_out[0], 1)
            t_point_w = (np.matmul(pose, t_point_cam))[:3]
            if not np.isnan(t_point_w[0]) and not np.isnan(t_point_w[1]):
                points_3d_corr.append(tuple(t_point_w))

        d = np.sum(np.diff(points_3d_corr, axis=0), axis=1)
        d_zero = np.where(d==0)[0]
        if len(d_zero) > 0:
            points_3d_corr = np.delete(points_3d_corr, d_zero, axis=0)

        if len(points_3d_corr) > 3:
            points_3d_corr_s = utils.compute_spline_3D(points_3d_corr, smoothing=1e-5, num_points=50)
        else:
            points_3d_corr_s = points_3d_corr

        ###########################################################################################
        # COMPUTE BORDERS
        points_1, points_2 = utils.borders_from_center(np.array(points_3d_corr_s), diameter=self.diameter)
        arr1_corr = utils.projection(pose, points_1, self.camera_model, use_distortion=use_distortion, keep_invalid=False)
        arr2_corr = utils.projection(pose, points_2, self.camera_model, use_distortion=use_distortion, keep_invalid=False)



        if len(arr1_corr) > 3:
            arr1_corrs = utils.compute_spline(arr1_corr, smoothing=1e-5)
        else:
            arr1_corrs = arr1_corr

        if len(arr2_corr) > 3:
            arr2_corrs = utils.compute_spline(arr2_corr, smoothing=1e-5)
        else:
            arr2_corrs = arr2_corr

        # MERGE BORDERS SINGLE LIST HEAD-TAIL-TAIL-HEAD
        list_points = []
        list_points.extend(arr1_corrs)
        list_points.extend(arr2_corrs[::-1])

        mask_out = utils.draw_mask_poly_fill(list_points, img.shape[:2])
        if cable is None:
            img_out_path = os.path.join(self.output_path, str(id).zfill(4) + ".png")
        else:
            img_out_path = os.path.join(self.output_path, str(id).zfill(4) + f"_{cable}.png")
        cv2.imwrite(img_out_path, mask_out)
        return 



    def process_image_init(self, id, pen_points, use_distortion=True):
        img_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_image.png")
        pose_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_pose.txt")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        pose = np.loadtxt(pose_path)

        # PROJECTION
        points_2d_input = utils.projection(pose, pen_points, self.camera_model, use_distortion=use_distortion, keep_invalid=True)

        # CORRECTION WITH NEURAL NETWORK
        points_2d_dict = {it: points_2d_input[it] for it, p in enumerate(pen_points) if points_2d_input[it] is not None}
        points_2d_corrected_dict = self.corrector_net.exec(points_2d_dict, img=img, debug=False)


        ###########################################################################################
        # PROJECT CORRECTED POINTS IN 3D FOR COMPUTING DLO BORDERS
        for it, pp in enumerate(pen_points):
            point_2d = points_2d_corrected_dict.get(it)
            if point_2d is not None:

                t_w = np.append(pp, 1)
                t_cam = (np.matmul(np.linalg.inv(pose), t_w))[:3]

                points_out = utils.unproject(np.array([point_2d]), np.array([t_cam[2]]), self.camera_model, use_distortion=use_distortion)

                t_point_cam = np.append(points_out[0], 1)
                t_point_w = (np.matmul(pose, t_point_cam))[:3]
                self.pen_points_dict[it].append(tuple(t_point_w))

        return


    def process_image_spline(self, id, pen_points, cable=None, use_distortion=True):

            # LOAD IMG AND POSE
            img_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_image.png")
            pose_path = os.path.join(self.imgs_path, str(id).zfill(4) + "_pose.txt")

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            pose = np.loadtxt(pose_path)

            ###########################################################################################
            # COMPUTE BORDERS
            points_1, points_2 = utils.borders_from_center(np.array(pen_points), diameter=self.diameter)
            arr1 = utils.projection(pose, points_1, self.camera_model, use_distortion=use_distortion, keep_invalid=False)
            arr2 = utils.projection(pose, points_2, self.camera_model, use_distortion=use_distortion, keep_invalid=False)

            # MERGE BORDERS SINGLE LIST HEAD-TAIL-TAIL-HEAD
            list_points = []
            list_points.extend(arr1)
            list_points.extend(arr2[::-1])


            mask_out = utils.draw_mask_poly_fill(list_points, img.shape[:2])
            if cable is None:
                img_out_path = os.path.join(self.output_path, str(id).zfill(4) + ".png")
            else:
                img_out_path = os.path.join(self.output_path, str(id).zfill(4) + f"_{cable}.png")
            cv2.imwrite(img_out_path, mask_out)

            return 


    def one_step_spline_run(self, cable=None, num_points=30):
        print("---> 1 step!")
        paths = glob.glob(os.path.join(self.imgs_path, "*.png"))
        data = [int(f.split("/")[-1].split("_")[0]) for f in paths]
        data = sorted(data)

        cprint("data loaded!", "yellow")

        self.pen_points_smoothed = utils.compute_spline_3D(self.pen_points, smoothing=0.5, num_points=num_points)

        for d in tqdm(data):
            self.process_image_spline(d, self.pen_points_smoothed, cable=cable)



    def one_step_run(self, cable=None, debug=False, num_points=30):
        print("---> 1 step!")
        paths = glob.glob(os.path.join(self.imgs_path, "*.png"))
        data = [int(f.split("/")[-1].split("_")[0]) for f in paths]
        data = sorted(data)

        cprint("data loaded!", "yellow")

        if len(self.pen_points) > 3:
            self.pen_points_smoothed = utils.compute_spline_3D(self.pen_points, smoothing=0.5, num_points=num_points)
        else:
            self.pen_points_smoothed = self.pen_points

        for d in tqdm(data):
            #if d % 5 == 0:
            self.process_image_end(d, self.pen_points_smoothed, cable=cable)


