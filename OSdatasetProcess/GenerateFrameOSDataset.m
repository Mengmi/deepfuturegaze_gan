%%
clear all;
clc;
close all;
%%
Config.ipath = 'OSdataset/videos/';
folderlist = dir(Config.ipath);
Config.opath = 'OSdataset/frames/';
Config.osize = [640 480];
%framesize = [640 480];
%Config.fsize = 40;
%vdir = dir([Config.ipath '/videos/*.mp4']);
%%
%action = {};
% 7 TO 15 TO 30 TO 45 TO 57
% error in video 5 (A13)
for vv=8:15%length(folderlist)
    display(['processing ' num2str(vv)]);
    vname = sprintf('%03d',vv-2);
    if ~exist([Config.opath vname '/'],'dir')
         mkdir([Config.opath vname '/']);    
    end
    
    %%
    foldername = folderlist(vv).name;
    vfile = [Config.ipath foldername '/'];
    videopath = dir([vfile '*.avi']);
    videoname = videopath(1).name;
    %evtfilename = dir([vfile '*.txt']);
    %%
    
    xyloObj = VideoReader([vfile videoname]);
    fRate = xyloObj.FrameRate;
    nFrames = xyloObj.NumberOfFrames;
    
    for j = 1:nFrames
        img = read(xyloObj,j);
        img = imresize(img, [128 128]);
        imwrite(img, [Config.opath vname '/frame_' num2str(j) '.png']);
    end
    
end
