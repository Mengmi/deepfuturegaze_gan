function [evt, raw, header] = read_smp_data(smp_file)
% smp_file - name of sample file
% Output:
% imgEvt: A eye-gaze structure, see below for details.
% evt: Raw data from idf file
% header: header information from idf file
%
% evt.fixation: x-position, y-position
% evt.saccade: x-position, y-position
% evt.blink: x-position, y-position
%


fp=fopen(smp_file,'r');

if fp == -1
    error('cannot open SMI txt file');
    evt = {};
    return
else
    display('openning file ... ');
end
%%
clear header
for ii=1:34
    header{ii} = fgetl(fp);
end

%%
raw = [];
evt.fixation = [];
evt.saccade = [];
evt.blink = [];
evt.stime = 0;
%%
if strcmp(header{4}, '## Version:	BeGaze 3.1.77')
    while (~feof(fp))
        %%
        str=fgetl(fp);
        %%
        [token remain] = strtok(str);
        raw{end+1}.time = sscanf(token,'%d');
        [token remain] = strtok(remain);
        raw{end}.type = token;
        [token remain] = strtok(remain);
        raw{end}.trial = sscanf(token,'%d');
        [token remain] = strtok(remain);
        raw{end}.px = sscanf(token,'%f');
        [token remain] = strtok(remain);
        raw{end}.py = sscanf(token,'%f');
        [token remain] = strtok(remain);
        raw{end}.frame = sscanf(token,'%d');
        [token remain] = strtok(remain);
        raw{end}.event = token;
        %%        
        if evt.stime == 0
            evt.stime = raw{end}.time;
        end
        
        %%
        switch raw{end}.event
            case 'Fixation'
               evt.fixation(end+1,:) = [raw{end}.px raw{end}.py raw{end}.time-evt.stime ]; 
            case 'Saccade'
               evt.saccade(end+1,:) = [raw{end}.px raw{end}.py raw{end}.time-evt.stime ]; 
            case 'Blink'
               evt.blink(end+1) = raw{end}.time-evt.stime; 
        end
        
    end
else
    err('Incorrect Version','Incorrect Version. BeGaze 3.1.77 Expected.');
end

%%
fclose(fp);