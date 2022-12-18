% data_file = 'labeledDomain_FEM_Beads_s1.json';
data_file = 'Confocal_segment_300.json';
% labeled bead domain
file = fopen(data_file);
raw = fread(file,inf);
str = char(raw');
fclose(file);
val = jsondecode(str);
% get struct field names
struct_names = fieldnames(val);
bead_numbers = fieldnames(val.(struct_names{end}));
% domain range
for i = 1 : length(struct_names)
    if strcmp(struct_names{i}, 'domain_size')
        shape = val.(struct_names{i});
        shape = shape(:)';
        domain = [0, shape(1), 0, shape(2), 0, shape(3)];
        break;
    end
end
% dx
for i = 1 : length(struct_names)
    if strcmp(struct_names{i}, 'voxel_size')
        dx = val.(struct_names{i});
        break;
    end
end
% number of beads
for i = 1 : length(struct_names)
    if strcmp(struct_names{i}, 'bead_count')
        num_beads = val.(struct_names{i});
        if num_beads ~= length(bead_numbers)
            error('.json data file: bead_count does not match number of beads listed')
        end
        break;
    end
end

% number of voxels and beads
% Shape        
nVPDx = (domain(2) - domain(1)) / dx;
nVPDy = (domain(4) - domain(3)) / dx;
nVPDz = (domain(6) - domain(5)) / dx;
nVoxels = nVPDx * nVPDy * nVPDz;

assert(mod(nVPDx, 1) == 0);
assert(mod(nVPDy, 1) == 0);
assert(mod(nVPDz, 1) == 0);

shape = [nVPDx, nVPDy, nVPDz];

% Create 3D grid of voxel coordinates (center of voxel cube)
voxels = centeredGrid3D(domain, dx);

% get bead data
bead_struct = struct;
bead_struct.Beads = cell(num_beads, 1);
remove_beads = false(num_beads, 1);
for i = 1 : num_beads
    ranges = val.(struct_names{end}).(bead_numbers{i});
    bead_struct.Beads{i} = cell2mat(arrayfun(@(x) (ranges(x, 1):ranges(x, end))', ...
                                             1 : size(ranges, 1), 'UniformOutput', false)');
    if isempty(bead_struct.Beads{i})
        remove_beads(i) = true;
    end
end
bead_struct.Beads(remove_beads) = [];
num_beads = num_beads - sum(remove_beads);

%%
voxel_count = cellfun(@length, bead_struct.Beads);
vol = zeros(nVoxels,1,'logical');
for i = 1 : num_beads
    if voxel_count(i)>10 % && voxel_count(i)<1000
        vol(bead_struct.Beads{i}) = 1;
    end
end
vol = reshape(vol,shape);
% figure; imshow3D(vol,[0,1]);


%% Create 3D grid of centered voxel coordinates
function voxelCenter = centeredGrid3D(bounds, dx)
 
    xMin = bounds(1);
    xMax = bounds(2);
    yMin = bounds(3);
    yMax = bounds(4);
    zMin = bounds(5);
    zMax = bounds(6);
    
    [xxx, yyy, zzz] = ndgrid(xMin : dx : xMax, ...
                             yMin : dx : yMax, ...
                             zMin : dx : zMax);
                      
    xxx = xxx(1 : end-1, 1 : end-1, 1 : end-1) + (dx / 2);
    yyy = yyy(1 : end-1, 1 : end-1, 1 : end-1) + (dx / 2);
    zzz = zzz(1 : end-1, 1 : end-1, 1 : end-1) + (dx / 2);
    voxelCenter = [xxx(:), yyy(:), zzz(:)];
 
end
