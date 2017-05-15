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

actionpath = '../adversialmask/video';
fileID = fopen('../adversial/fulllistMask_train.txt','w');

for i =[2 3 5 20]%6:8 10 12:14 16:18 21:22 2 3 5 20]% 
    
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
                %img = imread(imgname);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%% GazeLocation %%%%%%%%%%%%%%%%
                [x1, y1, x2, y2] = getGazeLocation_Tobii(i, k); 
                Ibinary = zeros(480,640);

                if (x1 == 0 && y1 == 0 )
                    x1 = x2;
                    y1 = y2;
                    %warning('there are zeros2');
                end

                if (x2 == 0 && y2 == 0 )
                    x2 = x1;
                    y2 = y1;
                    %warning('there are zeros3');
                end


                
                if x1 >0 && x1< 640 && y1 > 0 && y1<480 && x2 >0 && x2< 640 && y2 > 0 && y2<480
                     Ibinary(y1,x1) = 1;
                     Ibinary(y2,x2) = 1;
                end

               
                G = fspecial('gaussian',[500 500],50);            
                Ig = imfilter(Ibinary,G,'same');
                Ig = mat2gray(Ig);
%                 imshow(Ig);
%                 drawnow;
%                 pause;
                img = Ig;
                %%%%%%%%%%%%%%%%%%%%%%%     END     %%%%%%%%%%%%%%%%%%%%%
                
                img = imresize(img, [imgsizew imgsizew]);

                if counter <= NumFrames
                   imgtotal = [imgtotal;img];

                   if counter == NumFrames
                      imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                      fprintf(fileID,'%s\n',['adversialmask/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
                   end

                else
                   imgtotal = imgtotal( (imgsizew+1):end, 1:imgsizew,:);
                   imgtotal = [imgtotal;img];
                   imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                   fprintf(fileID,'%s\n',['adversialmask/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
                end
                counter = counter +1;

            end
        else
            
            for k = startFrame: endFrame

                downpngstr = sprintf('%010d', k);
                imgname = ['../png/' upngstr '/' downpngstr '.png'];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%%%%% GazeLocation %%%%%%%%%%%%%%%%
                [x1, y1, x2, y2] = getGazeLocation_Tobii(i, k);
                
               
        %         if x1 == 0 || y1 == 0 || x2 == 0 || y2 == 0
        %             continue;
        %         end

                

                Ibinary = zeros(480,640);

                if (x1 == 0 && y1 == 0 )
                    x1 = x2;
                    y1 = y2;
                    %warning('there are zeros2');
                end

                if (x2 == 0 && y2 == 0 )
                    x2 = x1;
                    y2 = y1;
                    %warning('there are zeros3');
                end

%                 if x1<1 ||y1<1 ||x2<1 ||y2<1
%                     %warning('there are zeros4');
%                     continue;
%                 end
% 
%                 if x1>hh ||y1>ww ||x2>hh ||y2>ww
%                     %warning('there are zeros5');
%                     continue;
%                 end
                
                if x1 >0 && x1< 640 && y1 > 0 && y1<480 && x2 >0 && x2< 640 && y2 > 0 && y2<480
                     Ibinary(y1,x1) = 1;
                     Ibinary(y2,x2) = 1;
                end

               
                G = fspecial('gaussian',[500 500],50);            
                Ig = imfilter(Ibinary,G,'same');
                Ig = mat2gray(Ig);
%                 imshow(Ig);
%                 drawnow;
%                 pause;
                img = Ig;
                %%%%%%%%%%%%%%%%%%%%%%%     END     %%%%%%%%%%%%%%%%%%%%%
                img = imresize(img, [imgsizew imgsizew]);

                
               imgtotal = [imgtotal;img];

               if counter == (endFrame - startFrame + 1)
                  imwrite(imgtotal,[saveimgpath '/' num2str(counter) '.jpg' ]); 
                  fprintf(fileID,'%s\n',['adversialmask/video' upngstr '/ac' num2str(j) '/' num2str(counter) '.jpg']);
               end                
               counter = counter +1;

            end
        end
    end

end
fclose(fileID);