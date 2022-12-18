%% Load beads from nd2 file
file='C:\Users\baoyi\Downloads\KW4-4A9-YIGSR-TUJ1555_GFAP488_Phall647_20x_001.nd2';
% finfo = nd2finfo(file)
[~, ~, img3d, ~] = nd2read3d(file);
% figure; imshow3D(img3d,[0,2000]);

%% Resize the 3D image
[Lx,Ly,Lz] = size(img3d);
img2d = reshape(img3d,[],Lz);
% Remove nonuniform background along z direction
img3d = img3d - permute(prctile(img2d,1,1),[1,3,2]);

dx = 0.61; dy = 0.61; dz = 1.1; % voxel sizes in the original 3D image (unit: um)
dxyz = 2; % voxel size in the resized 3D image (unit: um)
Lx1 = round(Lx*dx/dxyz);
Ly1 = round(Ly*dy/dxyz);
Lz1 = round(Lz*dz/dxyz);
img3d_resize = imresize3(img3d,[Lx1,Ly1,Lz1]);

% figure; imshow3D(img3d_resize,[0,2000]);
save('Confocal_2_image.mat','img3d_resize','-v7')

