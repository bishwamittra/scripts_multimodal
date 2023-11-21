"""
Installation instructions for openslide
1. apt-get install openslide-tools
2. apt-get install python-openslide
3. pip install openslide-python

"""
import argparse
import collections
import os
import re
from glob import glob
from typing import Any, Dict, List

import openslide
from PIL import ImageDraw
from tqdm import tqdm


class VisualizeHeatMaps(object):
    """
    Helper class to plot TCGA heatmaps. Can be used in two different ways
    1. Plotting multiple heatmaps
        helper = VisualizeHeatMaps(data_dir, log_dir, output_dir)
        helper.plot(image_size, out_folder)
    2. Plotting heatmap of a single image
        prediction_info = VisualizeHeatMaps.parse_log_file(log_dir)[patient_id] # Get the prediction info for a pid
        VisualizeHeatMaps.plot_heatmap(image_path, prediction_info, image_size patient_id, plot_original) # Plot the heatmap
    """
    def __init__(self, data_dir: str, log_dir: str):
        """
        log_dir: Path to log files from predict run
        data_dir: Path to data where svs images are present
        """
        super().__init__()
        self.log_dir = log_dir
        self.data_dir = data_dir
        self.pid_to_predictions = VisualizeHeatMaps.parse_log_file(log_dir)
        self.pid_to_image_paths = VisualizeHeatMaps.get_image_paths(data_dir)

    def plot(self, image_size, out_folder):
        for pid in tqdm(self.pid_to_predictions.keys()):
            if pid not in self.pid_to_image_paths:
                print(f"Image unavailable for {pid}")
                continue
            self._plot(pid, image_size, out_folder)

    @staticmethod
    def plot_heatmap(image_path: str,
                     prediction_info: List[Dict[str, Any]],
                     image_size: int,
                     output_folder: str,
                     patient_id: str,
                     plot_original: bool = False):
        """
        Plots the heat map for a patient given the prediction information.
        
        Inputs:
            image_path: Path to the svs image
            prediction_info: Contains list of patch level prediction information parsed from prediction logs. 
                             Each entry is Dict containing coordinates and predicted value.
            image_size: Size of the patches.
            output_folder: Location to store the heatmap.
            patient_id: Used for file name.
            plot_original: Flag for plotting the original image along with the heatmap.
        """
        os.makedirs(output_folder, exist_ok=True)
        slide = openslide.OpenSlide(image_path)
        image_size *= float(slide.properties['openslide.objective-power']) // 20
        stride = 512 * float(slide.properties['openslide.objective-power']) // 20

        slide_x, slide_y = slide.dimensions
        downsample_x, downsample_y = slide.level_dimensions[-1]
        downsampled_image = slide.get_thumbnail((downsample_x, downsample_y))

        if plot_original:
            original_path = os.path.join(output_folder, f"{patient_id}_original.png")
            downsampled_image.save(original_path)

        x_factor = downsample_x / slide_x
        y_factor = downsample_y / slide_y

        draw = ImageDraw.Draw(downsampled_image, "RGBA")

        for log_line in prediction_info:
            start_x = int(log_line["x"]) * x_factor * stride - image_size * x_factor // 2
            start_y = int(log_line["y"]) * y_factor * stride - image_size * x_factor // 2

            end_x = start_x + image_size * x_factor
            end_y = start_y + image_size * y_factor
            shape = ((start_x, start_y), (end_x, end_y))
            color = (255, 0, 0, 125) if log_line["predicted"] == 1 else (0, 0, 255, 125)
            draw.rectangle(shape, fill=color)

        downsampled_image.save(os.path.join(output_folder, f"{patient_id}.png"))

    def _plot(self, pid, image_size, out_folder):
        image_path = self.pid_to_image_paths[pid]
        prediction_info = self.pid_to_predictions[pid]
        VisualizeHeatMaps.plot_heatmap(image_path, prediction_info, image_size, out_folder, pid)

    @staticmethod
    def parse_log_file(log_dir):
        """
        Helper function to compute the patient ids to patch level predictions.
        """
        logs = []

        # Read all logs
        for log_file in os.listdir(log_dir):
            logs.extend(VisualizeHeatMaps._read_log_file(os.path.join(log_dir, log_file)))

        pid_to_prediction_info = collections.defaultdict(list)
        for log in logs:
            pid_to_prediction_info[log["pid"]].append(log)

        return pid_to_prediction_info

    @staticmethod
    def get_image_paths(data_dir):
        pid_to_image = {}
        result = [y for x in os.walk(data_dir) for y in glob(os.path.join(x[0], '*.svs'))]

        for image_path in result:
            pid = image_path.split("/")[-1].split(".")[0]
            pid_to_image[pid] = image_path

        return pid_to_image

    @staticmethod
    def _read_log_file(log_file: str):
        is_log_line = lambda x: "pid:" in x
        logs = []
        with open(log_file) as f:
            for line in f.read().split("\n"):
                if is_log_line(line):
                    logs.append(VisualizeHeatMaps._parse_log_line(line))
        return logs

    @staticmethod
    def _parse_log_line(line):
        key_mapping = {"pid: ": "pid", "target: ": "target", "predicted: ": "predicted"}
        info = {}
        for key in key_mapping:
            try:
                info[key_mapping[key]] = float(re.split(r'(\s|\,)', line.split(key)[1])[0])
            except:
                info[key_mapping[key]] = re.split(r'(\s|\,)', line.split(key)[1])[0]

        patient_id = info["pid"].split(".")[0]
        info["pid"], info["x"], info["y"] = patient_id.split("_")
        return info


def add_arguments(parser):
    parser.add_argument('--log-dir', type=str, help="Path to log files from prediction runs")
    parser.add_argument('--data-dir', type=str, help="path to data dir where svs images exist")
    parser.add_argument('--output-dir', type=str, help="path to the output directory for heatmaps", default="./output/")
    parser.add_argument('--image-size', type=int, help="Size of the original patches", default=512)


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    # plot the heatmaps
    helper = VisualizeHeatMaps(args.data_dir, args.log_dir)
    helper.plot(args.image_size, args.output_dir)


if __name__ == '__main__':
    main()
