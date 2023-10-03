import pickle
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
import imageio as iio
import cv2

import matplotlib.pyplot as plt


def load_intrinsic_camera_calibration(intrinsic_camera_calibration_filepath: str) -> Tuple[np.array, np.array]:
    with open(intrinsic_camera_calibration_filepath, "rb") as io:
        intrinsic_calibration = pickle.load(io)
    return intrinsic_calibration


def undistort_points(df_raw, camera_parameters_for_undistortion: Dict) -> pd.DataFrame:
    # understanding the maths behind it: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
    points = df_raw[["x", "y"]].values
    points_undistorted = cv2.undistortPoints(
        points,
        camera_parameters_for_undistortion["K"],
        camera_parameters_for_undistortion["D"],   
    )
    points_undistorted = np.squeeze(points_undistorted)
    return points_undistorted, df_raw["likelihood"]


class DLCUndistorter:
    def __init__(self, dlc_filepath: str, intrinsic_camera_calibration_filepath: str, video_filepath: str):
        self.intrinsic_camera_calibration_filepath = intrinsic_camera_calibration_filepath
        if self.intrinsic_camera_calibration_filepath.endswith(".p"):
            self.intrinsic_camera_calibration = load_intrinsic_camera_calibration(self.intrinsic_camera_calibration_filepath)
        
        self.dlc_filepath = dlc_filepath
        if self.dlc_filepath.endswith("h5"):
            self.dlc_df = pd.read_hdf(dlc_filepath, header=[0, 1, 2], index_col=0)
        elif self.dlc_filepath.endswith("csv"):
            self.dlc_df = pd.read_csv(dlc_filepath, header=[0, 1, 2], index_col=0)
        else:
            raise ValueError("DeepLabCut file must be .h5 or .csv")
            
        self.video_filepath = video_filepath
        self.image = iio.v3.imread(self.video_filepath, index=0)
            
            
    def run(self) -> pd.DataFrame:
        size = self.image.shape[1], self.image.shape[0]
        camera_parameters_for_undistortion = {"K": self.intrinsic_camera_calibration["K"], "D": self.intrinsic_camera_calibration["D"], "size": size}
        
        scorer = self.dlc_df.columns.levels[0][0]
        bps = self.dlc_df.columns.levels[1]
        df_undistorted = pd.DataFrame({}, index=self.dlc_df.index, columns=self.dlc_df.columns)
        
        for bp in bps:
            xy, likelihood = undistort_points(self.dlc_df.loc[:, (scorer, bp)], camera_parameters_for_undistortion)
            df_undistorted.loc[:, (scorer, bp, "likelihood")] = likelihood
            df_undistorted.loc[:, (scorer, bp, "x")] = xy[:, 0]
            df_undistorted.loc[:, (scorer, bp, "y")] = xy[:, 1]

        return df_undistorted