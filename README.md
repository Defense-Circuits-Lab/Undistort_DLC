# Easy undistortion of DeepLabCut Data

DeepLabCut is a widely used markerless pose estimation toolbox in behavioral science. Removing camera distortion from fisheye and non-fisheye cameras is crucial for calculations on DeepLabCut data. Notably, the code in this repository is basic and an implementation of undistortion as an optional function in the DeepLabCut toolbox itself would be appreciated.

The Undistort_DLC repository takes care of this specific task: Take DeepLabCut dataframes as the input and output undistorted DeepLabCut dataframes. Requirements for this undistortion are an intrinsic camera calibration and the video belonging to the deeplabcut dataframe. It is possible, to take croppings of the video belonging to the deeplabcut dataframe in regard to the intrinsic camera calibration video, into account. The code is working on both fisheye and non-fisheye cameras.



This is a [Defense Circuits Lab](https://www.defense-circuits-lab.com/) project written by Konstantin Kobel.
<table>
<td>
    <a href="https://www.defense-circuits-lab.com/"> 
        <img src="https://static.wixstatic.com/media/547baf_87ffe507a5004e29925dbeb65fe110bb~mv2.png/v1/fill/w_406,h_246,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/LabLogo3black.png" alt="DefenseCircuitsLab" style="width: 250px;"/>
    </a>
</td>

</table>
