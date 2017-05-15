function [imgEvt evt header] = fcn_MMread_event_data(event_file)
% fixation_file - name of Events file
% Output:
% imgEvt: A eye-gaze structure, see below for details.
% evt: Raw data from idf file
% header: header information from idf file
%
% imgEvt.left: Events for left eye
% imgEvt.right: Events for right eye
% Both imgEvt.left and imgEvt.right have the same structure. See below for
% details
%
% imgEvt.[left|right].fixation: x-position, y-position, duration,
% pupil size (pupil_x * pupil_y), start time of fixation, end time of
% fixation
% imgEvt.[left|right].saccade: Duration, Amplitude,	Peak Speed, Peak Speed
% At time, Average Speed,	Peak Accel.,Peak Decel.,	Average Accel.,
% imgEvt.[left|right].blink: Duration, Start time, End time
%

%event_file = 'E:/OSdataset/videos/A1/Event Export_Participant 1_participant 1-3-recording.txt';
fp=fopen(event_file,'r');

if fp == -1
    imgEvt = {};
    return
end
%%
clear header
for ii=1:20
    header{ii} = fgetl(fp);
end

%%
evt = [];
imgEvt.left.fixation = [];
imgEvt.left.saccade = [];
imgEvt.left.blink = [];
imgEvt.right.fixation = [];
imgEvt.right.saccade = [];
imgEvt.right.blink = [];
stime = 0;
%%
if strcmp(header{4}, 'Version:	BeGaze 3.5.158') || strcmp(header{4}, 'Version:	BeGaze 3.5.90') ...
        || strcmp(header{4}, 'Version:	IDF Event Detector 3.0.17')
    while (~feof(fp))
        %%
        str=fgetl(fp);
        %%
        [token remain] = strtok(str);
        evt{end+1}.event = token;
        [token remain] = strtok(remain);
        evt{end}.type = token;
        evt{end}.raw = sscanf(remain,'%f');
        %%        
        if stime == 0
            if length(evt{end}.raw) == 2
                stime = evt{end}.raw(2);
                imgEvt.stime = stime;
            else
                stime = evt{end}.raw(3);
                imgEvt.stime = stime;
            end
        end
        
        switch evt{end}.event
            case 'Fixation'
                %Trial	Number	Start	End	Duration	Location X	Location Y	Dispersion X	Dispersion Y	Plane	Avg. Pupil Size X	Avg Pupil Size Y
                if evt{end}.type == 'L'
                    imgEvt.left.fixation(end+1,:) = [evt{end}.raw(6) evt{end}.raw(7) evt{end}.raw(5) evt{end}.raw(11)*evt{end}.raw(12) evt{end}.raw(3)-stime evt{end}.raw(4)-stime ];
                else
                    imgEvt.right.fixation(end+1,:) = [evt{end}.raw(6) evt{end}.raw(7) evt{end}.raw(5) evt{end}.raw(11)*evt{end}.raw(12) evt{end}.raw(3)-stime evt{end}.raw(4)-stime ];
                end
            case 'Saccade'
                %Trial	Number	Start	End	Duration	Start Loc.X	Start Loc.Y	End Loc.X	End Loc.Y	Amplitude	Peak Speed	Peak Speed At	Average Speed	Peak Accel.	Peak Decel.	Average Accel.
                %Duration Amplitude	Peak Speed	Peak Speed At	Average Speed	Peak Accel.	Peak Decel.	Average Accel.
                if evt{end}.type == 'L'
                    imgEvt.left.saccade(end+1,:) = [evt{end}.raw(5) evt{end}.raw(10:end)' evt{end}.raw(3)-stime evt{end}.raw(4)-stime];
                else
                    imgEvt.right.saccade(end+1,:) = [evt{end}.raw(5) evt{end}.raw(10:end)' evt{end}.raw(3)-stime evt{end}.raw(4)-stime];
                end
            case 'Blink'
                %Trial	Number	Start	End	Duration
                if evt{end}.type == 'L'
                    imgEvt.left.blink(end+1,:) = [evt{end}.raw(5) evt{end}.raw(3)-stime evt{end}.raw(4)-stime];
                else
                    imgEvt.right.blink(end+1,:) = [evt{end}.raw(5) evt{end}.raw(3) evt{end}.raw(4)];
                end
        end
        
    end
else
    err('Incorrect Version','Incorrect Version. BeGaze 3.5.74 Expected.');
end

%%
fclose(fp);

end %end of function
