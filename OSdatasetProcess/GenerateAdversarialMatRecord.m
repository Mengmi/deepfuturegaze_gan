%% Mengmi Zhang %%
%% Date: June 6th, 2016%%
%% Topic: Action Recognition for Egocentric Video%%
clear all; close all; clc;

%ActionList = {'take','open','scoop','spread','close','sandwich','pour'};
% 15 is not a good sequence
% 4, 9, 11 and 19 are missing files
% 2,3,5,20 for testing
% for i ,1825 images


NumFrames = 40;
imgsizew = 128;


cellssave = cell(5706,NumFrames);
%6510
%14084
%8260
%4674
%7703

reccol = 1;
recrow = 1;

load('OStable.mat');

Config.ipath = './VXY/';
framesize = [640 480];
vdir = dir([Config.ipath '*.mat']);


for i = 1:43 
    
    
    vname = sprintf('%03d',OStable(i,1));
    
    load([Config.ipath  'gaze_' num2str(OStable(i,1)) '.mat']);
    
    startFrame = OStable(i,2);
    endFrame = OStable(i,3);
    acflag = 1;

    counter = 1;          
        
    for k = startFrame: endFrame
        structsave = [];

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%% GazeLocation %%%%%%%%%%%%%%%%
        temp = vxy{k};

        if ~isempty(temp)
            vx = ceil(temp(1,:)/framesize(1)*640);
            vy = ceil(temp(2,:)/framesize(2)*480);

            vy(find(vx<1)) = [];
            vx(find(vx<1)) = [];

            vy(find(vx>framesize(1))) = [];
            vx(find(vx>framesize(1))) = [];

            vx(find(vy<1)) = [];
            vy(find(vy<1)) = [];

            vx(find(vy>framesize(2))) = [];
            vy(find(vy>framesize(2))) = [];

            y1 = ceil(mean(vy));
            y2 = y1;

            x1 = ceil(mean(vx));
            x2 = x1;


            if x1 >=1 && x1<= 640 && y1 >=1 && y1<=480 && x2 >=1 && x2<= 640 && y2 >=1 && y2<=480
                 flag = 1;
            else
                flag = 0;
            end
        else
            flag = 0;
        end               


        if counter <= NumFrames
           structsave.video = OStable(i,1);
           structsave.frame = k;
           structsave.x1 = x1;
           structsave.y1 = y1;
           structsave.x2 = x2;
           structsave.y2 = y2;
           structsave.flag =flag;
           structsave.acflag = acflag;

           cellssave{recrow,reccol} = structsave;

           reccol = reccol + 1;


           if counter == NumFrames
               recrow = recrow + 1;
               reccol = 1;

           end

        else

           cellssave(recrow,1:(NumFrames-1)) = cellssave(recrow-1,2:NumFrames);
           reccol =  NumFrames;
           structsave.video = OStable(i,1);
           structsave.frame = k;
           structsave.x1 = x1;
           structsave.y1 = y1;
           structsave.x2 = x2;
           structsave.y2 = y2;
           structsave.flag =flag; 
           structsave.acflag = acflag;

           cellssave{recrow,reccol} = structsave;
           recrow = recrow + 1;
           reccol = 1;

        end
        counter = counter +1;

    end   

end

save(['../Code/GTAdversarial_os.mat'],'cellssave');
display('done saving');
