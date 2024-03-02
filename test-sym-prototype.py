import math
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

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


nusc = NuScenes(version='v1.0-trainval', dataroot='./v1.0-trainval09_blobs', verbose=True)

print("------------------- All Scenes -------------------")
nusc.list_scenes()
print("--------------------------------------------------")

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(15, 12))
plt.show(block=False)
plt.connect('key_press_event', on_key)

fig.canvas.mpl_connect('close_event', on_close)

auto_skip = True

for scene in nusc.scene:
    print("Scene Data:", scene, "\n")

    sample_token = scene['first_sample_token']

    log_token = scene["log_token"]
    log = nusc.get("log", log_token)

    for x in range(scene["nbr_samples"]):
        sample = nusc.get('sample', sample_token)

        print("Sample Data:", sample, "\n")

        cam_front_name = 'CAM_FRONT'
        if (log["vehicle"] == "n015") :
            cam_front_right_name = 'CAM_FRONT_LEFT'
        else:
            cam_front_right_name = 'CAM_FRONT_RIGHT'
        lidar_name = "LIDAR_TOP"

        cam_data = nusc.get('sample_data', sample['data'][cam_front_name])
        print("Cam Data:", cam_data, "\n")

        lidar_data = nusc.get('sample_data', sample['data'][lidar_name])
        print("LIDAR Data:", lidar_data, "\n")

        print("Car Name:", log["vehicle"], "\n")

        try:
            pc = LidarPointCloud.from_file("v1.0-trainval09_blobs/" + lidar_data["filename"])
        except FileNotFoundError as e:
            print(f"Skipping LidarPointCloud file {lidar_data['filename']} because it does not exist.")
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

        _, boxes, _ = nusc.get_sample_data(lidar_data["token"])

        car_boxes = [box for box in boxes if box.name.startswith('vehicle.car') or box.name.startswith('vehicle.truck')]
        car_boxes_tokens = []

        for box in car_boxes:
            annotation = nusc.get("sample_annotation", box.token)
            if len(annotation['attribute_tokens']) == 0:
                car_boxes_tokens.append([])
                continue
            temp_list = []
            for att_token in annotation['attribute_tokens']:
                att = nusc.get("attribute", att_token)
                temp_list.append(att)
            car_boxes_tokens.append(temp_list)

        car_boxes_filtered = []
        
        for i in range(len(car_boxes)):
            for token_x in car_boxes_tokens[i]:
                if token_x["name"] == 'vehicle.parked':
                    car_boxes_filtered.append(car_boxes[i])

        ax3.clear()

        for i in range(len(car_boxes_filtered)):
            if (log["vehicle"] == "n015"):
                car_boxes_filtered[i].center[0] = -car_boxes_filtered[i].center[0]
                car_boxes_filtered[i].orientation.q[1] = -car_boxes_filtered[i].orientation.q[1]
                car_boxes_filtered[i].orientation.q[3] = -car_boxes_filtered[i].orientation.q[3]

        for i in range(len(car_boxes_filtered)):
            distances = []

            box1 = car_boxes_filtered[i]
            c1 = np.array(nusc.colormap[box1.name]) / 255.0

            if not box1.center[0] > 0 or not box1.center[1] > 0:
                continue

            if box1.center[0] > 10:
                continue

            box1.render(ax3, view=np.eye(4), colors=(c1, c1, c1))

            for j in range(len(car_boxes_filtered)):
                box2 = car_boxes_filtered[j]
                c2 = np.array(nusc.colormap[box2.name]) / 255.0

                if not box2.center[0] > 0 or not box2.center[1] > 0:
                    continue
                
                if box2.center[0] > 10:
                    continue

                if box2.center[1] <= box1.center[1]:
                    continue

                distance = math.sqrt((box1.center[0] - box2.center[0]) ** 2 + (box1.center[1] - box2.center[1]) ** 2) - (box1.wlh[1]/2) - (box2.wlh[1]/2)

                distances.append([box2.center, distance])

            distances = sorted(distances, key=lambda x: x[1])

            margin = 2

            if len(distances) > 0:
                shortest = 0

                for d in distances:
                    x1 = box1.center[0]-margin
                    x2 = box1.center[0]+margin
                    o = d[0][0]
                    if (o > x1) and (o < x2):
                        shortest = d
                        break

                if shortest == 0:
                    continue
                if shortest[1] > 5 and shortest[1] < 30:
                    auto_skip = False

                # Display the distance as a line
                random_color = np.random.rand(3,)
                line = np.array([box1.center[:2], shortest[0][:2]]).T
                ax3.plot(line[0], line[1], color=random_color, linestyle='dashed')

                # Display the distance value near the midpoint of the line
                midpoint = 0.5 * (line[:, 0] + line[:, 1])
                ax3.text(midpoint[0], midpoint[1], f'{shortest[1]:.2f}', color='black', fontsize=8, ha='center', va='center')

        if (not auto_skip):
            # Display front camera on left
            cam_data_front = nusc.get('sample_data', sample['data'][cam_front_name])
            image_front= Image.open("v1.0-trainval09_blobs/" + cam_data_front["filename"])

            if (log["vehicle"] == "n015") :
                image_array = np.array(image_front)
                symmetric_image_array = np.flip(image_array, axis=1)
                symmetric_image = Image.fromarray(symmetric_image_array)
                image_front = symmetric_image
            
            results_front = model(image_front, classes=[2, 5, 7])
            for r_front in results_front:
                im_array_front = r_front.plot()
                im_front = Image.fromarray(im_array_front[..., ::-1])
            
            ax1.clear()
            ax1.imshow(im_front)
            #ax1.imshow(image_front)
            ax1.axis('off')

            # Display front-right camera on middle
            cam_data_front_right = nusc.get('sample_data', sample['data'][cam_front_right_name])
            image_front_right = Image.open("v1.0-trainval09_blobs/" + cam_data_front_right["filename"])

            if (log["vehicle"] == "n015") :
                image_array = np.array(image_front_right)
                symmetric_image_array = np.flip(image_array, axis=1)
                symmetric_image = Image.fromarray(symmetric_image_array)
                image_front_right = symmetric_image
            
            results_front_right = model(image_front_right, classes=[2, 5, 7])
            for r_front_right in results_front_right:
                im_array_front_right = r_front_right.plot()
                im_front_right = Image.fromarray(im_array_front_right[..., ::-1])
            
            ax2.clear()
            ax2.imshow(im_front_right)
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