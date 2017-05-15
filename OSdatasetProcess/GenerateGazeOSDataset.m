%%
clear all;
clc;
close all;
%%
Config.ipath = './videos/';
folderlist = dir(Config.ipath);
Config.opath = './VXY/';
Config.osize = [1280 960];
%framesize = [640 480];
%Config.fsize = 40;
%vdir = dir([Config.ipath '/videos/*.mp4']);
%%
action = {};
for vv=1:length(folderlist)
    
    display(['processing #' num2str(vv)]);
    %%
    vname = sprintf('%03d',vv-2);
    foldername = folderlist(vv).name;
    vfile = [Config.ipath foldername '/'];
    videopath = dir([vfile '*.avi']);
    videoname = videopath(1).name;
    evtfilename = dir([vfile '*.txt']);
    %%
    
    xyloObj = VideoReader([vfile videoname]);
    fRate = xyloObj.FrameRate;
    nFrames = xyloObj.NumberOfFrames;
    
    %%
    efile = [vfile evtfilename(1).name];
    evtdata = fcn_MMread_event_data(efile);
    fdata = evtdata.right.fixation;
      
    vx = ceil(fdata(:,1) * (xyloObj.Width/Config.osize(1)));
    vy = ceil(fdata(:,2) * (xyloObj.Height/Config.osize(2)));
    %center = [median(vx) median(vy)];
    
    %%
    %av = nan(nFrames,1);
    vxy = cell(nFrames,1);
    %%
    for ii = 1 : nFrames
        %%
        ctime = [(ii-1)/fRate*10^6 ii/fRate*10^6];
        gv = find(ctime(1) <= fdata(:,5) & fdata(:,5) <= ctime(2));
          
        if ~isempty(gv)
            
            vxy{ii} = [vx(gv),vy(gv)]';
            
            %%to test fixations and plot
%             Ibinary = zeros(xyloObj.Height, xyloObj.Width);
%             img = imread(['E:\OSdataset\frames\' vname '\frame_' num2str(ii) '.png' ]);
%             Ibinary(vy(gv), vx(gv)) = 1;
%             G = fspecial('gaussian',[300 300],50);
%             Ibinary = imfilter(Ibinary,G,'same');
%             imshow(heatmap_overlay(img, Ibinary));
%             pause(1);
%             drawnow;
            
        end
       
    end
    %%av stores the corresponding row number in SMI.txt for each frame;
    %%i.e. the action
    %%vxy stores the corresponding gaze for each frame
    %%adata stores the starting and ending time for each action
    save([Config.opath '/gaze_' num2str(vv-2) '.mat'],'vxy');
            
end
