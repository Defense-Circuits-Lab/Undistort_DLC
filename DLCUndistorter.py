import pickle
from typing import Tuple, Union, Dict, Optional

import warnings
import numpy as np
import pandas as pd
import imageio as iio
import cv2

import matplotlib.pyplot as plt


def load_intrinsic_camera_calibration(intrinsic_camera_calibration_filepath: str, cropping: Optional[Dict] = None) -> Tuple[np.array, np.array]:
    with open(intrinsic_camera_calibration_filepath, "rb") as io:
        intrinsic_calibration = pickle.load(io)

    if cropping is not None:
        intrinsic_calibration["K"][0][2] = intrinsic_calibration["K"][0][2] - cropping["offset_col_idx"]
        intrinsic_calibration["K"][1][2] = intrinsic_calibration["K"][1][2] - cropping["offset_row_idx"]
        
    return intrinsic_calibration


def undistort_points(df_raw, camera_parameters_for_undistortion: Dict, fisheye: bool = False) -> pd.DataFrame:
    # understanding the maths behind it: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html
    points = df_raw[["x", "y"]].values

    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    
    if fisheye:
        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_parameters_for_undistortion["K"],
            camera_parameters_for_undistortion["D"], 
            camera_parameters_for_undistortion["size"], 
            None, 
            balance=0
        )
        points_undistorted = cv2.fisheye.undistortPoints(
            np.expand_dims(points, axis=1),
            camera_parameters_for_undistortion["K"], 
            camera_parameters_for_undistortion["D"], 
            None, 
            newcameramtx
            )
    
    else:
        points_undistorted = cv2.undistortPoints(
            points,
            camera_parameters_for_undistortion["K"],
            camera_parameters_for_undistortion["D"],   
        )
        
    points_squeezed = np.squeeze(points_undistorted)
        
    return points_squeezed, df_raw["likelihood"]


class DLCUndistorter:
    def __init__(self, dlc_filepath: str, intrinsic_camera_calibration_filepath: str, video_filepath: str, fisheye: bool = False, cropping: Optional[Dict] = None):
        self.intrinsic_camera_calibration_filepath = intrinsic_camera_calibration_filepath
        if self.intrinsic_camera_calibration_filepath.endswith(".p"):
            self.intrinsic_camera_calibration = load_intrinsic_camera_calibration(self.intrinsic_camera_calibration_filepath, cropping=cropping)
        
        self.dlc_filepath = dlc_filepath
        if self.dlc_filepath.endswith("h5"):
            self.dlc_df = pd.read_hdf(dlc_filepath, header=[0, 1, 2], index_col=0)
        elif self.dlc_filepath.endswith("csv"):
            self.dlc_df = pd.read_csv(dlc_filepath, header=[0, 1, 2], index_col=0)
        else:
            raise ValueError("DeepLabCut file must be .h5 or .csv")
            
        self.video_filepath = video_filepath
        self.image = iio.v3.imread(self.video_filepath, index=0)
        
        self.fisheye = fisheye
            
            
    def run(self) -> pd.DataFrame:
        size = self.image.shape[1], self.image.shape[0]
        camera_parameters_for_undistortion = {"K": self.intrinsic_camera_calibration["K"], "D": self.intrinsic_camera_calibration["D"], "size": size}
        
        scorer = self.dlc_df.columns.levels[0][0]
        bps = self.dlc_df.columns.levels[1]
        df_undistorted = pd.DataFrame({}, index=self.dlc_df.index, columns=self.dlc_df.columns)
        
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        
        for bp in bps:
            xy, likelihood = undistort_points(self.dlc_df.loc[:, (scorer, bp)], camera_parameters_for_undistortion, self.fisheye)
            df_undistorted.loc[:, (scorer, bp, "likelihood")] = likelihood
            df_undistorted.loc[:, (scorer, bp, "x")] = xy[:, 0]
            df_undistorted.loc[:, (scorer, bp, "y")] = xy[:, 1]

        return df_undistorted
