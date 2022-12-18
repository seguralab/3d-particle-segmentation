% modified from https://github.com/joe-of-all-trades/nd2read
% Copyright (c) 2019 Joe Yeh
 
function [im_ch1, im_ch2, im_ch3, im_ch4] = nd2read3d(filename, varargin)
tic
finfo = nd2finfo(filename);
disp(['analyzing file structure used ', sprintf('%0.2f', toc), ' seconds'])
Lx = finfo.img_width;
Ly = finfo.img_height;
Lz = finfo.img_seq_count;
 
if finfo.ch_count == 4
    im_ch1 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch2 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch3 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch4 = zeros(Lx, Ly, Lz, 'uint16');
elseif finfo.ch_count == 3
    im_ch1 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch2 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch3 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch4 = uint16(0);
elseif finfo.ch_count == 2
    im_ch1 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch2 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch3 = uint16(0);
    im_ch4 = uint16(0);
elseif finfo.ch_count == 1
    im_ch1 = zeros(Lx, Ly, Lz, 'uint16');
    im_ch2 = uint16(0);
    im_ch3 = uint16(0);
    im_ch4 = uint16(0);
end
 
tic
fid = fopen(filename, 'r');
for z = 1:Lz
    fseek(fid, finfo.file_structure(contains({finfo.file_structure(:).nameAttribute},...
        ['ImageDataSeq|',num2str(z-1),'!'])).dataStartPos, 'bof');
 
    % Image extracted from ND2 has image width defined by its first dimension.
    if finfo.padding_style == 1
        if finfo.ch_count == 4
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch3(:, ii, z) = temp(1, :);
                im_ch1(:, ii, z) = temp(2, :);
                im_ch2(:, ii, z) = temp(3, :);
                im_ch4(:, ii, z) = temp(4, :);
                fseek(fid, 2, 'cof');
            end
        elseif finfo.ch_count == 3
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch3(:, ii, z) = temp(1, :);
                im_ch1(:, ii, z) = temp(2, :);
                im_ch2(:, ii, z) = temp(3, :);
                fseek(fid, 2, 'cof');
            end
        elseif finfo.ch_count == 2
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch3(:, ii, z) = temp(1, :);
                im_ch1(:, ii, z) = temp(2, :);
                fseek(fid, 2, 'cof');
            end
        elseif finfo.ch_count == 1
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch3(:, ii, z) = temp(1, :);
                fseek(fid, 2, 'cof');
            end
        end
    else
        if finfo.ch_count == 4
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch1(:, ii, z) = temp(1, :);
                im_ch2(:, ii, z) = temp(2, :);
                im_ch3(:, ii, z) = temp(3, :);
                im_ch4(:, ii, z) = temp(4, :);
            end
        elseif finfo.ch_count == 3
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch1(:, ii, z) = temp(1, :);
                im_ch2(:, ii, z) = temp(2, :);
                im_ch3(:, ii, z) = temp(3, :);
            end
        elseif finfo.ch_count == 2
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch1(:, ii, z) = temp(1, :);
                im_ch2(:, ii, z) = temp(2, :);
            end
        elseif finfo.ch_count == 1
            for ii = 1: finfo.img_height
                temp = reshape(fread(fid, finfo.ch_count * finfo.img_width, '*uint16'),...
                    [finfo.ch_count finfo.img_width]);
                im_ch1(:, ii, z) = temp(1, :);
            end
        end
    end
end
fclose(fid);
 
if finfo.ch_count == 4
    im_ch1 = permute(im_ch1, [2,1,3]);
    im_ch2 = permute(im_ch2, [2,1,3]);
    im_ch3 = permute(im_ch3, [2,1,3]);
    im_ch4 = permute(im_ch4, [2,1,3]);
elseif finfo.ch_count == 3
    im_ch1 = permute(im_ch1, [2,1,3]);
    im_ch2 = permute(im_ch2, [2,1,3]);
    im_ch3 = permute(im_ch3, [2,1,3]);
elseif finfo.ch_count == 2
    im_ch1 = permute(im_ch1, [2,1,3]);
    im_ch2 = permute(im_ch2, [2,1,3]);
elseif finfo.ch_count == 1
    im_ch1 = permute(im_ch1, [2,1,3]);
end
% if any(strcmpi(varargin, 'use_ch4'))
%     im_ch3 = im_ch4;
% end
  
 
disp(['reading complete image data used ', sprintf('%0.2f', toc), ' seconds'])
end
