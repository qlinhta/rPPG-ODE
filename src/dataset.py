"""import os
import cv2
import argparse
import numpy as np
from moviepy.editor import VideoFileClip
import logging
from colorama import Fore, Style, init
import warnings
import shutil

warnings.filterwarnings("ignore")

init()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_to_mp4(input_dir: str, output_dir: str) -> None:
    for subject in os.listdir(input_dir):
        subject_path = os.path.join(input_dir, subject)
        if os.path.isdir(subject_path):
            avi_path = os.path.join(subject_path, 'vid.avi')
            if os.path.exists(avi_path):
                mp4_path = os.path.join(output_dir, subject, 'vid.mp4')
                os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
                logging.info(Fore.GREEN + f"Converting {avi_path} to {mp4_path}" + Style.RESET_ALL)
                clip = VideoFileClip(avi_path)
                clip.write_videofile(mp4_path, codec='libx264', audio=False, verbose=False)
                ground_truth_src = os.path.join(subject_path, 'ground_truth.txt')
                ground_truth_dst = os.path.join(output_dir, subject, 'ground_truth.txt')
                if os.path.exists(ground_truth_src):
                    os.makedirs(os.path.dirname(ground_truth_dst), exist_ok=True)
                    shutil.copy(ground_truth_src, ground_truth_dst)
                    convert_ground_truth_to_numpy(ground_truth_dst, subject, output_dir)


def convert_ground_truth_to_numpy(ground_truth_path: str, subject: str, output_dir: str) -> None:
    ground_truth = np.loadtxt(ground_truth_path)
    heart_rate = ground_truth[0, :]  # Taking the first row
    output_path = os.path.join(output_dir, f"{subject}_gt", "heart_rate.npy")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, heart_rate)
    logging.info(Fore.GREEN + f"Saved ground truth for {subject} to {output_path}" + Style.RESET_ALL)


def extract_frames(input_dir: str, output_dir: str, fps: int) -> None:
    for subject in os.listdir(input_dir):
        subject_path = os.path.join(input_dir, subject)
        if os.path.isdir(subject_path):
            mp4_path = os.path.join(subject_path, 'vid.mp4')
            if os.path.exists(mp4_path):
                vidcap = cv2.VideoCapture(mp4_path)
                success, image = vidcap.read()
                frame_count = 0
                while success:
                    timestamp = int(vidcap.get(cv2.CAP_PROP_POS_MSEC))
                    frame_path = os.path.join(output_dir, subject, f"{timestamp}_frame{frame_count}.jpg")
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    cv2.imwrite(frame_path, image)
                    logging.info(Fore.BLUE + f"Extracted frame {frame_count} at {timestamp} ms from {mp4_path}" + Style.RESET_ALL)
                    success, image = vidcap.read()
                    frame_count += 1
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, (frame_count * 1000 // fps))


def main(input_dir: str, output_dir: str, fps: int) -> None:
    logging.info(Fore.YELLOW + f"STARTING VIDEO CONVERSION // CONVERTING FROM AVI TO MP4" + Style.RESET_ALL)
    convert_to_mp4(input_dir, output_dir)
    logging.info(Fore.YELLOW + f"STARTING FRAME EXTRACTION // EXTRACTING FRAMES AT {fps} FPS" + Style.RESET_ALL)
    extract_frames(output_dir, output_dir, fps)
    logging.info(Fore.GREEN + f"DONE! PROCESSED DATA SAVED IN {output_dir}" + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UBFC dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the original dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the processed data will be saved.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second to extract from the videos.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.fps)
"""

import os
import cv2
import argparse
import numpy as np
from moviepy.editor import VideoFileClip
import logging
from colorama import Fore, Style, init
import warnings

warnings.filterwarnings("ignore")

init()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def convert_to_mp4(input_dir: str, output_dir: str) -> None:
    bin_dir = os.path.join(output_dir, 'bin')
    os.makedirs(bin_dir, exist_ok=True)
    for subject in os.listdir(input_dir):
        subject_path = os.path.join(input_dir, subject)
        if os.path.isdir(subject_path):
            avi_path = os.path.join(subject_path, 'vid.avi')
            if os.path.exists(avi_path):
                mp4_path = os.path.join(bin_dir, f'{subject}.mp4')
                logging.info(Fore.GREEN + f"Converting {avi_path} to {mp4_path}" + Style.RESET_ALL)
                clip = VideoFileClip(avi_path)
                clip.write_videofile(mp4_path, codec='libx264', audio=False, verbose=False)
                ground_truth_src = os.path.join(subject_path, 'ground_truth.txt')
                if os.path.exists(ground_truth_src):
                    convert_ground_truth_to_numpy(ground_truth_src, subject, output_dir)


def convert_ground_truth_to_numpy(ground_truth_path: str, subject: str, output_dir: str) -> None:
    ground_truth = np.loadtxt(ground_truth_path)
    heart_rate = ground_truth[0, :]  # Taking the first row
    output_path = os.path.join(output_dir, f"{subject}_gt.npy")
    np.save(output_path, heart_rate)
    logging.info(Fore.GREEN + f"Saved ground truth for {subject} to {output_path}" + Style.RESET_ALL)


def extract_frames(input_dir: str, output_dir: str, fps: int) -> None:
    bin_dir = os.path.join(output_dir, 'bin')
    for subject in os.listdir(bin_dir):
        mp4_path = os.path.join(bin_dir, subject)
        if os.path.isfile(mp4_path) and mp4_path.endswith('.mp4'):
            subject_name = os.path.splitext(subject)[0]
            vidcap = cv2.VideoCapture(mp4_path)
            frames = []
            success, image = vidcap.read()
            while success:
                frames.append(image)
                success, image = vidcap.read()
            frames_array = np.array(frames)
            output_path = os.path.join(output_dir, f"{subject_name}_frames.npy")
            np.save(output_path, frames_array)
            logging.info(Fore.BLUE + f"Extracted and saved frames for {subject_name}" + Style.RESET_ALL)
            vidcap.release()


def main(input_dir: str, output_dir: str, fps: int) -> None:
    logging.info(Fore.YELLOW + "STARTING VIDEO CONVERSION // CONVERTING FROM AVI TO MP4" + Style.RESET_ALL)
    convert_to_mp4(input_dir, output_dir)
    logging.info(Fore.YELLOW + f"STARTING FRAME EXTRACTION // EXTRACTING FRAMES AT {fps} FPS" + Style.RESET_ALL)
    extract_frames(output_dir, output_dir, fps)
    logging.info(Fore.GREEN + f"DONE! PROCESSED DATA SAVED IN {output_dir}" + Style.RESET_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess UBFC dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the original dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the processed data will be saved.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second to extract from the videos.")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.fps)
