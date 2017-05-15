%% Mengmi Zhang %%
%% Date: June 6th, 2016%%
%% Topic: Action Recognition for Egocentric Video%%
clear all; close all; clc;

%ActionList = {'take','open','scoop','spread','close','sandwich','pour'};
% 15 is not a good sequence
% 4, 9, 11 and 19 are missing files
% 2,3,5,20 for testing
NumFrames = 60;
imgsizew = 128;

actionpath = '../adversial/video';
fileID = fopen('../adversial/fulllist.txt','w');

for i =[1 6:8 10 12:14 16:18 21:22 2 3 5 20]% 
    
    ActionLabel = parseLabels(i);
     
    upngstr = sprintf('%03d', i);
    display(['dataset: ' upngstr]);
    
    command = sprintf('mkdir %s', [actionpath upngstr]);
    system(command);
    
    for j = 1: length(ActionLabel)
        
        struct = ActionLabel{j};
        startFrame = ActionLabel{j}.startFrame;
        endFrame = ActionLabel{j}.endFrame;
        
        saveimgpath =[actionpath upngstr '/ac' num2str(j)]; 
        command = sprintf('mkdir %s', saveimgpath);
        system(command);
        
        counter = 1;
        imgtotal = [];
        
        if (endFrame - startFrame + 1) >= NumFrames
        
            for k = startFrame: endFrame

                downpngstr = sprintf('%010d', k);
                imgname = ['../png/' upngstr '/' downpngstr '.png'];
                img = imread(imgname);
                img = imresize(img, [imgsizew imgsizew]);

                if counter <= NumFrames
                   imgtotal = [imgtotal;img];

                   if counter == NumFrames
                      imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                      fprintf(fileID,'%s\n',['adversial/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
                   end

                else
                   imgtotal = imgtotal( (imgsizew+1):end, 1:imgsizew,:);
                   imgtotal = [imgtotal;img];
                   imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                   fprintf(fileID,'%s\n',['adversial/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
                end
                counter = counter +1;

            end
        else
            
            for k = startFrame: endFrame

                downpngstr = sprintf('%010d', k);
                imgname = ['../png/' upngstr '/' downpngstr '.png'];
                img = imread(imgname);
                img = imresize(img, [imgsizew imgsizew]);

                
               imgtotal = [imgtotal;img];

               if counter == (endFrame - startFrame + 1)
                  imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                  fprintf(fileID,'%s\n',['adversial/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
               end                
               counter = counter +1;

            end
        end
    end

end
fclose(fileID);