#!/usr/bin/python

import argparse
import math
import statistics
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator
from scipy.spatial.transform import Rotation


SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 800.0
IMG_WIDTH = 1241
IMG_HEIGHT = 376
ANN_BOX_SIZE = 25
ANN_BOX_X_OFF = 15
ANN_BOX_Y_OFF = 20

MIN_ALLOWED_DEPTH = 5
MAX_ALLOWED_DEPTH = 20

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple scale consistency checker for ORB-SLAM2 and Monodepth2')

    parser.add_argument('--trajectory', type=str,
                        help='path to a trajectory file produced by ORB-SLAM2 (TUM format)', required=True)
    parser.add_argument('--keypoints', type=str,
                        help='path to a keypoints file produced by ORB-SLAM2 (custom format)', required=True)
    parser.add_argument('--sequence', type=str,
                        help='path to a KITTI secuence', required=True)
    parser.add_argument('--monodepth', type=str,
                        help='path to a Monodepth2 repository in file system', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_model",
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    
    parser.add_argument('--start', type=int, 
                        help='trajectory frame to start from', default=0)
    parser.add_argument('--stop', type=int,
                        help='trajectory frame to stop after', default=-1)
    parser.add_argument('--step', type=int,
                        help='frames to skip', default=1)

    return parser.parse_args()

def interpret_TUM(s):
    fields = s.strip().split(' ')
    timestamp = fields[0]
    translation = np.array(list(map(float, fields[1:4]))).reshape(3,1)
    quaternion = list(map(float, fields[4:8]))
    rotation = Rotation.from_quat(quaternion).as_matrix()
    change_of_base = np.append(np.append(rotation, translation, axis = 1),  [[0, 0, 0, 1]], axis=0)
    return {timestamp: change_of_base}

def read_trajectory(args):
    with open(args.trajectory) as f:
        trajectory = dict()
        for s in f.readlines()[args.start:args.stop:args.step]:
            trajectory.update(interpret_TUM(s))
        return trajectory


def read_keypoints(args):
    with open(args.keypoints) as f:
        lines = f.readlines()
        n = 0
        frames = dict()
        while n < len(lines):
            split = lines[n].split(' ')
            n = n + 1
            if len(split) == 3:
                [img, timestamp, _] = split
                keypoints = []
                while n < len(lines):
                    split = lines[n].strip().split(' ')
                    if len(split) == 5:
                        n = n + 1
                        keypoints.append(tuple(map(float, split)))
                    else:
                        break
                frames[timestamp] = [img, keypoints]
        return frames

def depth_cam_to_point(change_of_base, x, y, z):
    point = np.array([[x], [y], [z], [1]])
    point_ = change_of_base @ point
    # print (point[2], point_[2])
    return point_[2]

def get_orb_depth(args, trajectory, keypoints):
    depths = dict()
    for (timestamp, change_of_base) in trajectory.items():
        if timestamp in keypoints:
            frame = keypoints[timestamp]
            # print(change_of_base)
            # print(change_of_base[:3, 3])
            depths[frame[0]] =  \
                [((w, h), depth_cam_to_point(change_of_base, x, y, z)) \
                    for (w, h, x, y, z) in frame[1]]
    return depths

class Dict2Obj(object):
    """
    Turns a dictionary into a class
    """
    #----------------------------------------------------------------------
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])

def get_mono_depth(args, trajectory, keypoints):
    # Code from "test_simple.py"
    #print(str(sys.path.insert(0, os.path.join(args.monodepth,  "test_simple.py"))))
    #test_simple = __import__(str(os.path.join(args.monodepth,  "test_simple.py")))
    image_0 = os.path.join(args.sequence, "image_0")
    png_count = len([f for f in os.listdir(image_0) if os.path.splitext(f)[1] == '.png'])
    npy_count = len([f for f in os.listdir(image_0) if os.path.splitext(f)[1] == '.npy'])
    if npy_count != png_count:
        sys.path.insert(0, args.monodepth)
        test_simple = __import__("test_simple", fromlist=["test_simple.py"])
        test_simple_args = {'image_path' : str(image_0),
                            'model_name' : args.model_name,
                            'ext' : 'png',
                            'no_cuda' : False}
        print(Dict2Obj(test_simple_args))
        test_simple.test_simple(Dict2Obj(test_simple_args))
    depths = dict()
    total = len(trajectory)
    count = 1
    for timestamp in trajectory:
        if timestamp in keypoints:
            frame = keypoints[timestamp]
            img = frame[0]
            disp_path = str(img).zfill(6) + '_disp.npy'
            disp_path = os.path.join(image_0, disp_path)

            pred_disp = np.load(disp_path).squeeze()
            pred_disp = cv2.resize(pred_disp, (IMG_WIDTH, IMG_HEIGHT))
            pred_depth = 1 / pred_disp
            pred_depth *= SCALE_FACTOR
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            sys.stdout.write("\rLoading depths... %d / %d" %(count, total))
            sys.stdout.flush()
            depths[img] = [((w, h), pred_depth[int(h)][int(w)]) \
                for (w, h, _, _, _) in frame[1]]
            count = count + 1
    return depths

def get_data_from_trajectory(args, trajectory, keypoints, orb_depth, mono_depth):
    plot_label = []
    plot_min = []
    plot_Q1 = []
    plot_Q2 = []
    plot_Q3 = []
    plot_max = []

    plot_turns = []

    img_path = []

    last_vec3_z = np.array([0.0, 0.0, 1.0])
    for timestamp in trajectory:
        if timestamp in keypoints:
            frame = keypoints[timestamp]
            img = frame[0]
            scale = [mono[1] / orb[1] for mono, orb in zip(mono_depth[img], orb_depth[img]) \
                if MIN_ALLOWED_DEPTH <= orb[1] and orb[1] <= MAX_ALLOWED_DEPTH]
            # num_features = len(scale)
            # sorted_scale = np.sort(scale)
            plot_label.append(int(img))
            scale = list(scale)

            image_0 = os.path.join(args.sequence, "image_0")
            img_path_ = str(img).zfill(6) + '.png'
            img_path.append(os.path.join(image_0, img_path_))

            Q1 = np.quantile(scale, 0.25)
            Q2 = np.quantile(scale, 0.50)
            Q3 = np.quantile(scale, 0.75)
            RIC = (Q3 - Q1)
            Xmin = max(min(scale), Q1 - 1.5 * RIC)
            Xmax = min(max(scale), Q3 + 1.5 * RIC)

            vec3_z = trajectory[timestamp][0:3,2]
            turn = 1.0 - abs(np.dot(last_vec3_z, vec3_z))
            last_vec3_z = vec3_z
            plot_turns.append(turn)

            plot_min.append(Xmin)
            plot_Q1.append(Q1)
            plot_Q2.append(Q2)
            plot_Q3.append(Q3)
            plot_max.append(Xmax)

    turns_begin = []
    turns_end = []
    is_turn = False

    max_turn = max(plot_turns)
    plot_turns = list(map(lambda x: x / max_turn, plot_turns))
    for i, turn in enumerate(plot_turns):
        if not is_turn:
            # Increase if FALSE POSITIVE (noise as turn)
            if (turn > 0.1):
                j = i
                while (j > 0 and plot_turns[j] > 0.025):
                    j -= 1
                turns_begin.append(plot_label[j])
                is_turn = True
        else:
            if (turn < 0.025):
                turns_end.append(plot_label[i])
                is_turn = False
    # if is_turn:
    #     turns_end.append(int(keypoints[-1][0]))

    return plot_label, plot_min, plot_Q1, plot_Q2, plot_Q3, plot_max, \
           plot_turns, turns_begin, turns_end, img_path

def img_show(args, trajectory, keypoints, orb_depth, mono_depth):

    plot_label, plot_min, plot_Q1, plot_Q2, plot_Q3, plot_max, \
    plot_turn, turns_begin, turns_end, _ = \
        get_data_from_trajectory(args, trajectory, keypoints, orb_depth, mono_depth)

    once = True
    
    plt.figure(2)
    # top=0.963,
    # bottom=0.041,
    # left=0.0,
    # right=0.99,
    # hspace=0.175,
    # wspace=0.04 

    # Big tick locators
    big_major_locator = MultipleLocator(500)
    big_major_locator.MAXTICKS = 100
    big_minor_locator = MultipleLocator(100)
    big_minor_locator.MAXTICKS = 1000

    # Small tick locators
    small_major_locator = MultipleLocator(10)
    small_major_locator.MAXTICKS = 100
    small_minor_locator = MultipleLocator(1)
    small_minor_locator.MAXTICKS = 1000

    ###########################################################################
    # Turn detection                                                          #
    ###########################################################################
    ax = plt.subplot(6, 2, 2)
    ax.clear()
    ax.set_xlim(plot_label[0], plot_label[-1])
    # ax.set_title('Turn magnitude (0.0 to 1.0)')
    ax.xaxis.set_major_locator(big_major_locator)
    ax.xaxis.set_minor_locator(big_minor_locator)

    ax.set_title('Turn magnitude (0.0 to 1.0) / Interquartile range (IQR) / Box-and-whisker (full)')
    ax.plot(plot_label, plot_turn,'k', linewidth=1)

    # Shadow turn zones
    for t1, t2 in zip(turns_begin, turns_end):
        ax.axvspan(t1, t2, alpha=0.3, color='gray')

    ###########################################################################
    # Interquartile range (IQR)                                               #
    ###########################################################################
    ax = plt.subplot(6, 2, 4)
    ax.clear()
    # ax.set_title("Interquartile range (IQR)")
    ax.set_xlim(plot_label[0], plot_label[-1])
    ax.xaxis.set_major_locator(big_major_locator)
    ax.xaxis.set_minor_locator(big_minor_locator)

    plot_IQR = [x1 - x2 for (x1, x2) in zip(plot_Q3, plot_Q1)]
    ax.plot(plot_label, plot_IQR,'k', linewidth=0.5)

    # Shadow turn zones
    for t1, t2 in zip(turns_begin, turns_end):
        ax.axvspan(t1, t2, alpha=0.3, color='gray')

    ###########################################################################
    # Box-and-whiskers (full)                                                 #
    ###########################################################################
    ax = plt.subplot(3, 2, 4)
    ax.clear()
    ax.set_ylim(min(plot_min), max(plot_max))
    ax.set_xlim(plot_label[0], plot_label[-1])
    ax.xaxis.set_major_locator(big_major_locator)
    ax.xaxis.set_minor_locator(big_minor_locator)
    
    # ax.set_title('Box-and-whisker (full)')
    ax.plot(plot_label, plot_min, 'k', linewidth=0.5, linestyle=(0, (5, 10)))
    ax.plot(plot_label, plot_Q1,'k', linewidth=0.5)
    ax.plot(plot_label, plot_Q2, 'b', linewidth=0.5)
    ax.plot(plot_label, plot_Q3, 'k', linewidth=0.5)
    ax.plot(plot_label, plot_max, 'k', linewidth=0.5, linestyle=(0, (5, 10)))

    # Shadow turn zones
    for t1, t2 in zip(turns_begin, turns_end):
        ax.axvspan(t1, t2, alpha=0.3, color='gray')

    # Ratio normalization
    np_min_c = 0
    # np_max_c = max(abs(1.0 - max(plot_max)), abs(1.0 - min(plot_min)))
    np_max_c = 0.08

    # Box-and-whisker plot zoom
    BOX_N_WHISKER_WIDTH = 100
    box_and_whisker_page = 0

    static_bar1 = None
    static_bar2 = None
    static_bar3 = None
    for timestamp in trajectory:
        if timestamp in keypoints:
            frame = keypoints[timestamp]
            img = frame[0]
            image_0 = os.path.join(args.sequence, "image_0")
            img_path = str(img).zfill(6) + '.png'
            img_path = os.path.join(image_0, img_path)

            ###################################################################
            # ORB-SLAM2 features and calculated depths (Ground truth)         #
            ###################################################################            
            ax = plt.subplot(3, 2, 1)
            ax.clear()
            ax.set_title('ORB-SLAM2 Features')
            ax.axis('off')
            img_data = mpimg.imread(img_path)
            ax.imshow(img_data, cmap='gray')

            indexes = []
            for i, d in enumerate(orb_depth[img]):
                if MIN_ALLOWED_DEPTH <= d[1] and d[1] <= MAX_ALLOWED_DEPTH:
                    indexes.append(i)

            x = []
            y = []
            orb = []
            ann = np.zeros((IMG_HEIGHT // ANN_BOX_SIZE, IMG_WIDTH // ANN_BOX_SIZE))
            for d in np.array(orb_depth[img])[indexes]:
                Px = int(d[0][0])
                Py = int(d[0][1])
                x.append(Px)
                y.append(Py)
                orb.append(d[1])
                ann_box_y = Py // ANN_BOX_SIZE
                ann_box_x = Px // ANN_BOX_SIZE
                if ann[ann_box_y, ann_box_x] == 0:
                    if img_data[min(Py + ANN_BOX_Y_OFF, IMG_HEIGHT - 1), Px] < 0.5:
                        color = 'w'
                    else:
                        color = 'k'
                    ax.annotate("%.2f" % (d[1]),
                        xy=(Px, Py),
                        xytext=(Px - ANN_BOX_X_OFF, Py + ANN_BOX_Y_OFF),
                        color=color,
                        weight='bold',
                        size=7)
                    ann[ann_box_y - 1 : ann_box_y + 1, ann_box_x - 1 : ann_box_x + 1] = 1
            c = np.array(orb).squeeze()
            # c = (1.0 - (c - np.min(c)) / (np.max(c) - np.min(c))).squeeze()
            mynorm = plt.Normalize(vmin=MIN_ALLOWED_DEPTH, vmax=MAX_ALLOWED_DEPTH)
            sc = ax.scatter(x, y, s=10, marker='o', c=c, cmap='plasma', norm=mynorm)
            if static_bar1 == None:
                static_bar1 = plt.colorbar(sc)

            ###################################################################
            # Monodepth features and predicted depths                         #
            ###################################################################
            ax = plt.subplot(3, 2, 3)
            ax.clear()
            ax.set_title('Monodepth2 Features')
            ax.axis('off')
            ax.imshow(img_data, cmap='gray')
            
            mono = []
            ann = np.zeros((IMG_HEIGHT // ANN_BOX_SIZE, IMG_WIDTH // ANN_BOX_SIZE))
            for d in np.array(mono_depth[img])[indexes]:
                Px = int(d[0][0])
                Py = int(d[0][1])
                mono.append(d[1])
                ann_box_y = Py // ANN_BOX_SIZE
                ann_box_x = Px // ANN_BOX_SIZE
                if ann[ann_box_y, ann_box_x] == 0:
                    if img_data[min(Py + ANN_BOX_Y_OFF, IMG_HEIGHT - 1), Px] < 0.5:
                        color = 'w'
                    else:
                        color = 'k'
                    ax.annotate("%.2f" % (d[1]),
                        xy=(Px, Py),
                        xytext=(Px - ANN_BOX_X_OFF, Py + ANN_BOX_Y_OFF),
                        color=color,
                        weight='bold',
                        size=7)
                    ann[ann_box_y - 1 : ann_box_y + 1, ann_box_x - 1 : ann_box_x + 1] = 1
            c = np.array(mono).squeeze()
            # c = (1.0 - (c - np.min(c)) / (np.max(c) - np.min(c))).squeeze()
            mynorm = plt.Normalize(vmin=MIN_ALLOWED_DEPTH, vmax=MAX_ALLOWED_DEPTH)
            sc = ax.scatter(x, y, s=10, marker='o', c=c, cmap='plasma', norm=mynorm)
            if static_bar2 == None:
                static_bar2 = plt.colorbar(sc)

            ###################################################################
            # Ratio GT/Prediction (i.e. Monodepth2 / ORB-SLAM2)               #
            ###################################################################
            ax = plt.subplot(3, 2, 5)
            ax.clear()
            ax.set_title('Ratio: abs(1.0 - (ORB / Mono))')
            ax.axis('off')
            ax.imshow(img_data, cmap='gray')
            
            ratio = []
            for d_orb, d_mono in \
                zip(np.array(orb_depth[img])[indexes], np.array(mono_depth[img])[indexes]):
                # print(d_orb[1], d_mono[1])
                ratio.append(abs(1.0 - d_mono[1] / d_orb[1]))
            c = np.array(ratio).squeeze()
            mynorm = plt.Normalize(vmin=np_min_c, vmax=np_max_c)
            sc = ax.scatter(x, y, s=10, marker='o', c=c, cmap='plasma', norm=mynorm)
            if static_bar3 == None:
                static_bar3 = plt.colorbar(sc)

            ###################################################################
            # Box and whisker plot: min, Q1, Q2, Q3 and max values of ratio   #
            ###################################################################
            if (box_and_whisker_page * BOX_N_WHISKER_WIDTH - 1 < int(img)):

                min_img = box_and_whisker_page * BOX_N_WHISKER_WIDTH
                box_and_whisker_page += 1
                max_img = box_and_whisker_page * BOX_N_WHISKER_WIDTH - 1

                i = -1
                j = -1
                for k, label in enumerate(plot_label):
                    if (i < 0 and label >= min_img):
                        i = k
                    if (j < 0 and label > max_img):
                        j = k + 1
                        break
                else:
                    j = len(plot_label)

                ax = plt.subplot(3, 2, 6)
                ax.clear()
                ax.set_ylim(min(plot_min), max(plot_max))
                ax.set_xlim(min_img, max_img + 1)
                ax.set_title('Box-and-whisker (min, Q1, Q2, Q3 and max)')
                ax.xaxis.set_major_locator(small_major_locator)
                ax.xaxis.set_minor_locator(small_minor_locator)

                ax.plot(plot_label[i:j], plot_min[i:j], 'k', linewidth=1, linestyle=(0, (5, 10)))
                ax.plot(plot_label[i:j], plot_Q1[i:j],'k', linewidth=1)
                ax.plot(plot_label[i:j], plot_Q2[i:j], 'b', linewidth=1)
                ax.plot(plot_label[i:j], plot_Q3[i:j], 'k', linewidth=1)
                ax.plot(plot_label[i:j], plot_max[i:j], 'k', linewidth=1, linestyle=(0, (5, 10)))

                for t1, t2 in zip(turns_begin, turns_end):
                    if (t1 < plot_label[j - 1] and t2 > plot_label[i]):
                        t1 = max(t1, plot_label[i])
                        t2 = min(t2, plot_label[j - 1])
                        ax.axvspan(t1, t2, alpha=0.3, color='gray')

            ###################################################################
            # Red progress line (for box-and-whisker plots)                   #
            ###################################################################
            ax = plt.subplot(6, 2, 2)
            line0 = ax.axvline(int(img), color='r')
            ax = plt.subplot(6, 2, 4)
            line1 = ax.axvline(int(img), color='r')
            ax = plt.subplot(3, 2, 4)
            line2 = ax.axvline(int(img), color='r')
            ax = plt.subplot(3, 2, 6)
            line3 = ax.axvline(int(img), color='r')

            plt.draw()
            if once:
                once = False
                plt.waitforbuttonpress()

            # plt.savefig('output/' + str(img).zfill(6) + '.png')
            plt.pause(0.01)

            line0.remove()
            line1.remove()
            line2.remove()
            line3.remove()

def turn_detection(args, trajectory, keypoints, orb_depth, mono_depth):

    plot_label, _, _, _, _, _, plot_turns, _, _, img_path = \
        get_data_from_trajectory(args, trajectory, keypoints, orb_depth, mono_depth)

    _, ((ax2), (ax1)) =  plt.subplots(2,1)
    ax1.plot(plot_label, plot_turns, 'k', linewidth=1)
    plt.draw()
    plt.waitforbuttonpress()

    i = 0
    for timestamp in trajectory:
        if timestamp in keypoints:
            frame = keypoints[timestamp]
            img = frame[0]

            ax2.clear()

            img_data = mpimg.imread(img_path[i])
            ax2.imshow(img_data, cmap='gray')
            i += 1

            line = ax1.axvline(int(img), color='r')

            plt.draw()
            plt.pause(0.01)

            line.remove()

def autolabel(ax, rects, format, colors = ["black"]):
    """Attach a text label above each bar in *rects*, displaying its height."""
    n = len(colors)
    for i, rect in enumerate(rects):
        height = rect.get_height()
        j = i % n
        c = colors[j]
        ax.annotate(format.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', color=c)

def box_and_whisker(args, trajectory, keypoints, orb_depth, mono_depth):

    plot_label, plot_min, plot_Q1, plot_Q2, plot_Q3, plot_max, \
    plot_turns, turns_begin, turns_end, _ = \
        get_data_from_trajectory(args, trajectory, keypoints, orb_depth, mono_depth)

    _, ((ax1), (ax2), (ax3)) =  plt.subplots(3,1)
    ax1.set_xlim(plot_label[0], plot_label[-1])
    ax2.set_xlim(plot_label[0], plot_label[-1])
    ax3.set_xlim(plot_label[0], plot_label[-1])

    ax1.set_title("Box-n-whisker (Min, Q1, Q2, Q3 and Max)")
    ax1.plot(plot_label, plot_min, 'k', linewidth=0.5, linestyle=(0, (5, 10)))
    ax1.plot(plot_label, plot_Q1,'k', linewidth=0.5)
    ax1.plot(plot_label, plot_Q2, 'b', linewidth=0.5)
    ax1.plot(plot_label, plot_Q3, 'k', linewidth=0.5)
    ax1.plot(plot_label, plot_max, 'k', linewidth=0.5, linestyle=(0, (5, 10)))

    ax2.set_title("Turn magnitude (Dot product, scaled 0.0 to 1.0)")
    ax2.plot(plot_label, plot_turns, 'k', linewidth=0.5)

    ax3.set_title("Interquartile range (IQR)")
    plot_IQR = [x1 - x2 for (x1, x2) in zip(plot_Q3, plot_Q1)]
    ax3.plot(plot_label, plot_IQR, 'k', linewidth=0.5)

    ###########################################################################
    # Statistic data                                                          #
    ###########################################################################

    error = []
    rmse = []
    RATIO_OR_ABSOLUTE = False # True ==> Ratio // False ==> Absolute error
    ratio = []
    MAX_HISTOGRAM = 100
    DIV_HISTOGRAM = 5
    histogram = [list() for _ in range(MAX_HISTOGRAM)]
    for (k, v) in orb_depth.items():
        orb = v
        mono = mono_depth[k]
        error.extend([abs(gt[1][0] - pred[1]) for gt, pred in zip(orb, mono) \
            if MIN_ALLOWED_DEPTH <= gt[1][0] and gt[1][0] <= MAX_ALLOWED_DEPTH])
        rmse.extend([(gt[1][0] - pred[1]) ** 2 for gt, pred in zip(orb, mono) \
            if MIN_ALLOWED_DEPTH <= gt[1][0] and gt[1][0] <= MAX_ALLOWED_DEPTH])
        ratio.extend([abs(1.0 - pred[1] / gt[1][0]) for gt, pred in zip(orb, mono) \
            if MIN_ALLOWED_DEPTH <= gt[1][0] and gt[1][0] <= MAX_ALLOWED_DEPTH])
        
        for gt, pred in zip(orb, mono):
            index = int(gt[1][0] // DIV_HISTOGRAM)
            if index < MAX_HISTOGRAM:
                if RATIO_OR_ABSOLUTE:
                    histogram[index].append(abs(1.0 - pred[1] / gt[1][0]))
                else:
                    histogram[index].append(abs(gt[1][0] - pred[1]))

    print('')
    print('-- Diff error (ORB - MONO) --')
    print('Mean error: ', np.mean(error))
    print('Median error: ', np.median(error))
    print('RMSE: ', math.sqrt(np.mean(rmse)))
    print('')

    print('-- Ratio: abs(1.0 - MONO / ORB) --')
    print('Mean ratio: ', np.mean(ratio))
    print('Min ratio:', min(ratio))
    print('Q1: ', np.quantile(ratio, 0.25))
    print('Median: ', np.median(ratio))
    print('Q3: ', np.quantile(ratio, 0.75))
    print('Max ratio:', max(ratio))
    print('')

    min_IQR = np.argmin(plot_IQR)
    print('Min IQR:', plot_IQR[min_IQR], ' (at frame ', plot_label[min_IQR] ,')')
    max_IQR = np.argmax(plot_IQR)
    print('Max IQR:', plot_IQR[max_IQR], ' (at frame ', plot_label[max_IQR] ,')')
    print('')

    _, ((ax4), (ax5)) =  plt.subplots(2,1)
    ax4.set_title("Nº muestras")
    ax5.set_title("Mediana del error %")
    x = []
    height_len = []
    height_median = []
    for i in range(0, len(histogram), 2):
        x.append(i * DIV_HISTOGRAM + DIV_HISTOGRAM / 2.0)
        median = np.median(histogram[i])
        height_len.append(len(histogram[i]))
        height_median.append(median)
    rects11 = ax4.bar(x, height_len, DIV_HISTOGRAM, color='tab:blue', label='even')
    rects12 = ax5.bar(x, height_median, DIV_HISTOGRAM, color='tab:blue', label='even')
    autolabel(ax4, rects11, '{}')
    autolabel(ax5, rects12, '{0:.2f}', ["black"])

    x = []
    height_len = []
    height_median = []
    for i in range(1, len(histogram), 2):
        x.append(i * DIV_HISTOGRAM + DIV_HISTOGRAM / 2.0)
        median = np.median(histogram[i])
        height_len.append(len(histogram[i]))
        height_median.append(median)
    rects21 = ax4.bar(x, height_len, DIV_HISTOGRAM, color='tab:orange', label='odd')
    rects22 = ax5.bar(x, height_median, DIV_HISTOGRAM, color='tab:orange', label='odd')
    # autolabel(ax4, rects21, '{}')
    # autolabel(ax5, rects22, '{0:.2f}')

    ###########################################################################
    # Find IQR local maxima                                                   #
    ###########################################################################

    max_iqr = []

    ## OLD VERSION ##

    # max_local_iqr = 0
    # max_local_i = -1
    # for i, iqr in enumerate(plot_IQR):
    #     if (max_local_i == -1 and iqr > 0.25):
    #         max_local_iqr = iqr
    #         max_local_i = i
    #     if (max_local_i != -1 and iqr > max_local_iqr):
    #         max_local_iqr = iqr
    #         max_local_i = i
    #     if (max_local_i != -1 and iqr < 0.15):
    #         max_iqr.append(max_local_i)
    #         max_local_i = -1

    search_step = len(plot_IQR) // 10
    for i in range(0, len(plot_IQR), search_step) :
        arg_max = np.argmax(plot_IQR[i : i + search_step])
        if (plot_IQR[arg_max + i] > 0.025):
            max_iqr.append(plot_label[arg_max + i])

    print('Some samples of HIGH IQR: ', max_iqr)
    for iqr in max_iqr:
        ax1.axvline(iqr, color='r')
        ax2.axvline(iqr, color='r')
        ax3.axvline(iqr, color='r')

    ###########################################################################
    # Turn zones                                                              #
    ###########################################################################

    for t1, t2 in zip(turns_begin, turns_end):
        ax1.axvspan(t1, t2, alpha=0.3, color='gray')
        ax2.axvspan(t1, t2, alpha=0.3, color='gray')
        ax3.axvspan(t1, t2, alpha=0.3, color='gray')
    
    # plt.xticks(np.arange(min(plot_label), max(plot_label)+1, 10.0))
    
    plt.show()
            
if __name__ == '__main__':
    args = parse_args()
    trajectory = read_trajectory(args)
    keypoints = read_keypoints(args)
    orb_depth = get_orb_depth(args, trajectory, keypoints)
    mono_depth = get_mono_depth(args, trajectory, keypoints)
    # img_show(args, trajectory, keypoints, orb_depth, mono_depth)
    # turn_detection(args, trajectory, keypoints, orb_depth, mono_depth)
    box_and_whisker(args, trajectory, keypoints, orb_depth, mono_depth)
    scale = dict()
    for (k, v) in orb_depth.items():
        orb = v
        mono = mono_depth[k]
        scale[k] = statistics.median(map(lambda x: x[0][1] / x[1][1], zip(orb, mono)))
        # if scale[k] > 400 and scale[k] < 500:
        #    print(k)
    # m = max(scale.values())
    # for k, v in scale.items():
    #     if v == m:
    #         print(k)
    # print(list(scale.values())[:5])
    
    quit()
    plt.figure(1)
    plt.plot(scale.keys(), scale.values())
    # plt.axis([0, 1, 0, 1])
            
