import math
from PIL import Image
import matplotlib
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.utils.geometry_utils import view_points
#%% sadas
#matplotlib.use('TkAgg')
# Load a model


def on_close(event):
    exit()

def on_key(event):
    global go_back, skip_sample, auto_skip
    if event.key == 'n':
        skip_sample = True
    if event.key == 'b':
        go_back = True
    if event.key == 'a':
        auto_skip = True
green = [0.1333333, 0.545, 0.1333333]
def draw_lot(ax1, box1, box2):#box1 arkadaki box2 öndeki araç
    corners1 = view_points(box1.corners(), camera_intrinsic, normalize=True)[:2, :]
    corners2 = view_points(box2.corners(), camera_intrinsic, normalize=True)[:2, :]
    """ax1.scatter(corners1[0], corners1[1], c='red', s=10, alpha=0.5)
    ax1.scatter(corners2[0], corners2[1], c='blue', s=10, alpha=0.5)"""
    if (log["vehicle"] == "n015") :
        ax1.plot([corners2.T[4][0], corners1.T[0][0]],
            [corners2.T[4][1], corners1.T[0][1]],
            color=green, linestyle="solid")
        ax1.plot([corners2.T[5][0], corners1.T[1][0]],
            [corners2.T[5][1], corners1.T[1][1]],
            color=green, linestyle="solid")
        ax1.plot([corners2.T[4][0], corners2.T[5][0]],
            [corners2.T[4][1], corners2.T[5][1]],
            color=green, linestyle="solid")
        ax1.plot([corners1.T[0][0], corners1.T[1][0]],
            [corners1.T[0][1], corners1.T[1][1]],
             color=green, linestyle="solid")
    else:
        ax1.plot([corners2.T[6][0], corners1.T[2][0]],
                [corners2.T[6][1], corners1.T[2][1]],
                color=green, linestyle="solid")
        ax1.plot([corners2.T[7][0], corners1.T[3][0]],
                [corners2.T[7][1], corners1.T[3][1]],
                color=green, linestyle="solid")
        ax1.plot([corners2.T[6][0], corners2.T[7][0]],
                [corners2.T[6][1], corners2.T[7][1]],
                color=green, linestyle="solid")
        ax1.plot([corners1.T[2][0], corners1.T[3][0]],
                [corners1.T[2][1], corners1.T[3][1]],
                color=green, linestyle="solid")

#%%
nusc = NuScenes(version='v1.0-trainval', dataroot='./v1.0-trainval01_blobs', verbose=True)

print("------------------- All Scenes -------------------")
nusc.list_scenes()
print("--------------------------------------------------")

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 12))
plt.show(block=False)
plt.connect('key_press_event', on_key)

fig.canvas.mpl_connect('close_event', on_close)

auto_skip = True

for scene in nusc.scene:
    #print("Scene Data:", scene, "\n")

    sample_token = scene['first_sample_token']

    log_token = scene["log_token"]
    log = nusc.get("log", log_token)

    for x in range(scene["nbr_samples"]):
        sample = nusc.get('sample', sample_token)

        #print("Sample Data:", sample, "\n")

        cam_front_name = 'CAM_FRONT'
        if (log["vehicle"] == "n015") :
            cam_front_right_name = 'CAM_FRONT_LEFT'
        else:
            cam_front_right_name = 'CAM_FRONT_RIGHT'
        lidar_name = "LIDAR_TOP"

        cam_data = nusc.get('sample_data', sample['data'][cam_front_name])
        #print("Cam Data:", cam_data, "\n")

        lidar_data = nusc.get('sample_data', sample['data'][lidar_name])
        #print("LIDAR Data:", lidar_data, "\n")

        #print("Car Name:", log["vehicle"], "\n")

        try:
            pc = LidarPointCloud.from_file("v1.0-trainval01_blobs/" + lidar_data["filename"])
        except FileNotFoundError as e:
            #print(f"Skipping LidarPointCloud file {lidar_data['filename']} because it does not exist.")
            continue
        pc_points = pc.points[:3, :]

        if (log["vehicle"] == "n015") :
            pc_points[0, :] = -pc_points[0, :]

        view = np.eye(4)

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = pc_points.shape[1]

        points = np.concatenate((pc_points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / 40 / np.sqrt(2))

        point_scale = 0.2
        cam_data_front = nusc.get('sample_data', sample['data'][cam_front_name])
        cam_data_front_right = nusc.get('sample_data', sample['data'][cam_front_right_name])
        _, boxes,_ = nusc.get_sample_data(lidar_data["token"])
        _, im_boxes, camera_intrinsic = nusc.get_sample_data(cam_data_front["token"])
        _, im_right_boxes, camera_intrinsic1 = nusc.get_sample_data(cam_data_front_right["token"])
        car_boxes = []
        im_front_boxes_list=[]
        im_front_right_boxes_list=[]
        for box in boxes:
            annotation = nusc.get("sample_annotation", box.token)
            if 'attribute_tokens' in annotation and annotation['attribute_tokens']:
                for att_token in annotation['attribute_tokens']:
                    att = nusc.get("attribute", att_token)
                    if att["name"]=="vehicle.parked":
                        #print("Degree: ", box.orientation.degrees)
                        #if box.name.startswith('vehicle.car') or box.name.startswith('vehicle.truck'):
                        car_boxes.append(box)
        for box in im_boxes:
            annotation = nusc.get("sample_annotation", box.token)
            if 'attribute_tokens' in annotation and annotation['attribute_tokens']:
                for att_token in annotation['attribute_tokens']:
                    att = nusc.get("attribute", att_token)
                    if att["name"]=="vehicle.parked":
                        #print("Degree: ", box.orientation.degrees)
                        #if box.name.startswith('vehicle.car') or box.name.startswith('vehicle.truck'):
                        im_front_boxes_list.append(box)
        for box in im_right_boxes:
            annotation = nusc.get("sample_annotation", box.token)
            if 'attribute_tokens' in annotation and annotation['attribute_tokens']:
                for att_token in annotation['attribute_tokens']:
                    att = nusc.get("attribute", att_token)
                    if att["name"]=="vehicle.parked":
                        #print("Degree: ", box.orientation.degrees)
                        #if box.name.startswith('vehicle.car') or box.name.startswith('vehicle.truck'):
                        im_front_right_boxes_list.append(box)

        ax1.clear()
        ax2.clear()
        ax3.clear()
        #Buraya tekrar bak
        if (log["vehicle"] == "n015"):
            for i in range(len(im_front_boxes_list)):
                im_front_boxes_list[i].center[0] = -im_front_boxes_list[i].center[0]
                im_front_boxes_list[i].orientation.q[1] = -im_front_boxes_list[i].orientation.q[1]
                im_front_boxes_list[i].orientation.q[3] = -im_front_boxes_list[i].orientation.q[3]
            for i in range(len(im_front_right_boxes_list)):
                im_front_right_boxes_list[i].center[0] = -im_front_right_boxes_list[i].center[0]
                im_front_right_boxes_list[i].orientation.q[1] = -im_front_right_boxes_list[i].orientation.q[1]
                im_front_right_boxes_list[i].orientation.q[3] = -im_front_right_boxes_list[i].orientation.q[3]
        for i in range(len(car_boxes)):
            if (log["vehicle"] == "n015"):
                car_boxes[i].center[0] = -car_boxes[i].center[0]
                car_boxes[i].orientation.q[1] = -car_boxes[i].orientation.q[1]
                car_boxes[i].orientation.q[3] = -car_boxes[i].orientation.q[3]

            distances = []

            box1 = car_boxes[i]
            c1 = np.array(nusc.colormap[box1.name]) / 255.0
            c3 = np.array(nusc.colormap["human.pedestrian.adult"]) / 255.0
            if not box1.center[0] > 0 or not box1.center[1] > 0:
                box1.render(ax3, view=np.eye(4), colors=(c3, c3, c3))
                #box1.render(ax1, view=camera_intrinsic, normalize=True, colors=(c3, c3, c3))
            elif box1.center[0] > 10:
                box1.render(ax3, view=np.eye(4), colors=(c3, c3, c3))
                #box1.render(ax1, view=camera_intrinsic, normalize=True, colors=(c3, c3, c3))
            else:
                box1.render(ax3, view=np.eye(4), colors=(c1, c1, c1))
                #box1.render(ax1, view=camera_intrinsic, normalize=True, colors=(c1, c1, c1))

            for j in range(len(car_boxes)):
                box2 = car_boxes[j]
                c2 = np.array(nusc.colormap[box2.name]) / 255.0

                if not box2.center[0] > 0 or not box2.center[1] > 0:
                    continue
                
                if box2.center[0] > 10:
                    continue

                if box2.center[1] <= box1.center[1]:
                    continue

                distance = math.sqrt((box1.center[0] - box2.center[0]) ** 2 + (box1.center[1] - box2.center[1]) ** 2) - (box1.wlh[1]/2) - (box2.wlh[1]/2)

                distances.append([distance, box2])

            distances = sorted(distances, key=lambda x: x[0])

            margin = 2

            if len(distances) > 0:
                shortest = 0

                for d in distances:
                    x1 = box1.center[0]-margin
                    x2 = box1.center[0]+margin
                    #print(d[1].center[0])
                    o = d[1].center[0]
                    if (o > x1) and (o < x2):
                        shortest = d
                        break

                if shortest == 0:
                    continue
                if 5 < shortest[0] < 30 and 80 < abs(box1.orientation.degrees) < 100 and 80 < abs(shortest[1].orientation.degrees) < 100:
                    auto_skip = False

                # Display the distance as a line
                random_color = np.random.rand(3,)
                #if box1.orientation.degrees 
                if 5 < shortest[0] < 30:
                    if 80 < abs(box1.orientation.degrees) < 100 and 80 < abs(shortest[1].orientation.degrees) < 100:
                        line = np.array([box1.center[:2], shortest[1].center[:2]]).T  
                        ax3.plot(line[0], line[1], color=random_color, linestyle='dashed')
                        for box3 in im_front_boxes_list:
                            if box3.token == box1.token:
                                for box4 in im_front_boxes_list:
                                    if box4.token==shortest[1].token:
                                        draw_lot(ax1, box3, box4),
                                        print(log["vehicle"])
                        for box3 in im_front_right_boxes_list:
                            if box3.token == box1.token:
                                for box4 in im_front_right_boxes_list:
                                    if box4.token==shortest[1].token:
                                        draw_lot(ax2, box3, box4)
                        
                        # Display the distance value near the midpoint of the line
                        #print("Box1:",box1.orientation.degrees)
                        #print("Box2:", shortest[2])

                        midpoint = 0.5 * (line[:, 0] + line[:, 1])
                        ax3.text(midpoint[0], midpoint[1], f'{shortest[0]:.2f}', color='black', fontsize=8, ha='center', va='center')
                    else:
                        print("Distance between 5 and 30 but Box1 isn't between 80-100. Box1:",box1.orientation.degrees)
                        print("Distance between 5 and 30 but Box1 isn't between 80-100.Box2:", shortest[1].orientation.degrees)


        if (not auto_skip):
            # Display front camera on left
            
            image_front= Image.open("v1.0-trainval01_blobs/" + cam_data_front["filename"])
            #_, boxes, camera_intrinsic = nusc.get_sample_data(cam_data_front["token"])
            if (log["vehicle"] == "n015") :
                image_array = np.array(image_front)
                symmetric_image_array = np.flip(image_array, axis=1)
                symmetric_image = Image.fromarray(symmetric_image_array)
                image_front = symmetric_image
            
            
            ax1.imshow(image_front)
            #ax1.imshow(image_front)
            ax1.axis('off')

                

            # Display front-right camera on middle
            cam_data_front_right = nusc.get('sample_data', sample['data'][cam_front_right_name])
            image_front_right = Image.open("v1.0-trainval01_blobs/" + cam_data_front_right["filename"])

            if (log["vehicle"] == "n015") :
                image_array = np.array(image_front_right)
                symmetric_image_array = np.flip(image_array, axis=1)
                symmetric_image = Image.fromarray(symmetric_image_array)
                image_front_right = symmetric_image
            
            
            ax2.imshow(image_front_right)
            #ax2.imshow(image_front_right)
            ax2.axis('off')

            # Display lidar right
            scatter = ax3.scatter(points[0, :], points[1, :], c=colors, s=point_scale)
            ax3.plot(0, 0, 'x', color='red')
            ax3.set_xlim(-40, 40)
            ax3.set_ylim(-40, 40)
            ax3.axis('off')
            ax3.set_aspect('equal')
            
            skip_sample = False
            go_back = False

            plt.waitforbuttonpress()

            if skip_sample:
                break
            
            if go_back and sample['prev'] != '':
                sample_token = sample['prev']
            else:
                sample_token = sample['next']
        else:
            sample_token = sample['next']