%% Load beads from SCAPE file
file='volumedata.mat';
load(file,'V','dx','dy','dz')
% figure; imshow3D(V,[0,600]);

%% Resize the 3D image
[Lx,Ly,Lz] = size(V);
dxyz = 2e-6; % voxel size in the resized 3D image (unit: m)
Lx1 = round(Lx*dx/dxyz);
Ly1 = round(Ly*dy/dxyz);
Lz1 = round(Lz*dz/dxyz);
img3d_resize = imresize3(V,[Lx1,Ly1,Lz1]);

% meanx = mean(mean(V,2),3);
% img3d_norm = uint16(double(V)./meanx*200);
% img3d_resize = imresize3(img3d_norm,[Lx1,Ly1,Lz1]);

% save('SCAPE_2_image.mat','img3d_resize','-v7')
% figure; imshow3D(img3d_resize,[0,600]);

%% Normalization in y direction v2
meanx = mean(mean(img3d_resize,2),3);
% To compensate for the nonuniform intensify at different yz plane, 
% divide the intensity of each yz plane by its mean intensity. 
img3d_norm = uint16(double(img3d_resize)./meanx*200);
img3d_resize = img3d_norm;
save('SCAPE_2_norm_image.mat','img3d_resize','-v7')
