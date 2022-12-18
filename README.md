This code used MATLAB and Python. Required Python packages: numpy, scipy, matplotlib, scikit-image, json.

MATLAB files: `resize_confocal.m` (calls `nd2read3d.m` and `nd2finfo.m`) resizes the confocal image to voxels size of 2 um in all directions, and compensates for nonuniform intensity in the z direction. `resize_SCAPE.m` resizes the SCAPE image to voxels size of 4 um in all directions, and compensates for nonuniform intensity in the x direction. Then they save the compensated and resized image in mat files (`{}_image.mat` in '-v7').

Python files: `segment3d.py` (calls `utils.py`) reads the resized image and segments to individual beads. It outputs labeled volume in mat files as well as the standard json file. Parameters for confocal or SCAPE images are set at the beginning of the script (one is commented). 

Main procedure includes: 
1. Threshold the 3D image to binary foreground. 
2. Calculate distance transform. 
3. Find initial watershed seeds, and eliminate seeds without significant prominence. 
4. Initial watershed.
5. Refine watershed seeds, and eliminate seeds that generate volumes with large surface-to-volume ratio. 
6. Second watershed.
7. Further cut one segmented region to multiple regions if possible.
8. Smooth the segmented regions by removing shart voxels.

Example original confocal image is available at [this link](https://www.icloud.com/attachment/?u=https%3A%2F%2Fcvws.icloud-content.com%2FB%2FAXFf37lvlridUV7g-E6_f5kf4EFzARGcfoHDbJNITjD66mkeHYxbZhCT%2F%24%7Bf%7D%3Fo%3DAnqP_SsaLUO7pB2d_7u3zmIi4g_xGKj3ZxYiuedViz9s%26v%3D1%26x%3D3%26a%3DCAoghKTpAKXhXXRfiNc5QOx4NG3I7PtmHjpoAFAPxKYfWXgSdBD4gbrfsy8Y-JG1s70vIgEAKgkC6AMA_xx_ZMpSBB_gQXNaBFtmEJNqJEPpKoZsmTNolZhBK2JkwJQO8WnQSBpLAyoaclr-3JdvgD8SE3Ik7pB9ras4hBdd119OQ7JZuI1SuVOmnrirJBECIRsEeJwsrBEW%26e%3D1631390091%26fl%3D%26r%3DD66525B9-516C-41B4-859C-9B873E94D0D4-1%26k%3D%24%7Buk%7D%26ckc%3Dcom.apple.largeattachment%26ckz%3DD846C956-D734-49E8-8FE8-DB3F376EC9C3%26p%3D34%26s%3DnEkg5Og3SDF0Kobo4Xl6WYS6MaM&uk=kTaaOwclQ0Ax6JFOyNJXxA&f=KW4-4A9-YIGSR-TUJ1555_GFAP488_Phall647_20x_001.nd2&sz=1201102848). Example original SCAPE image is `volumedata.mat`. Example results are saved in `results` folder. 

Tips for tuning parameters:
* If some very bright beads are incrrectly split, try decreasing `inten_max`.
* If some dark beads are ignored, try increasing `th`.
* If some dark gaps are incorrectly attributed to beads, try decreasing `th` or `th_relative`. 
* If some beads are incorrectly split, then investigate the possible reason. If the brightness is not uniform and has some relatively low-intensity local minima, try increasing `th_ralative`. If the brightness is relatively uniform, try increasing `peak_prom` or `d_peak`.
* If some beads are incorrectly merged, try decreaing `peak_prom` or `d_peak`
* If some beads have iregular shapes, try decreasing `s2v_max`.
* If some small beads are not identified, try inccreasing `s2v_max`.
* If the segmentation quality is bad on the edge of the image, consider ignore it if you can't improve it.
* Usually we cannot get perfect segmentation on all beads. Try manually merge some components if needed, or do a segmentation on a small volume near a problematic bead for that particular bead. 