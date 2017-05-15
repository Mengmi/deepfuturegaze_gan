function actions = parseLabels(seq)

% this paerses the action sequences (in Particular for Intel1)

filestr = sprintf('../labels/%03d.txt', seq);
fid = fopen(filestr);
actionStruct = struct('action', '', 'objects', [], 'seq', seq, 'startFrame', 0, 'endFrame', 0, 'periodic', false);
objectStruct = struct('object', '', 'seq', 0, 'startFrame', seq, 'endFrame', 0);
actions = cell(0);

while(true)
    linestr = fgetl(fid);
    if(~ischar(linestr))
        break;
    end
    if(strfind(linestr, '><'))
        %it is an action line
        linestr = linestr(2:end);
        [temp, remain] = strtok(linestr, '>');
        actionStruct.action = temp;
        remain = remain(3:end);
        [temp,remain] = strtok(remain, '>');
        actionStruct.objects = cell(0);
        while(length(temp) >= 1)
            [tt, temp] = strtok(temp, ',');
            temp = temp(2:end);
            actionStruct.objects{end+1} = tt;
        end
        remain = remain(4:end);
        [fromStr, remain] = strtok(remain, '-');
        actionStruct.startFrame = sscanf(fromStr, '%d');
        remain = remain(2:end);
        [endStr, remain] = strtok(remain, ')');
        actionStruct.endFrame = sscanf(endStr, '%d');
        remain = remain(4:end);
        actionStruct.periodic = sscanf(remain, '%d');
        actions{end+1} = actionStruct;
    else
        %it is an object line
        linestr = linestr(2:end);
        [objectStruct.object, remain] = strtok(linestr, '>');
        remain = remain(4:end);
        [fromStr, remain] = strtok(remain, '-');
        objectStruct.startFrame = sscanf(fromStr, '%d');
        remain = remain(2:end);
        endStr = strtok(remain, ')');
        objectStruct.endFrame = sscanf(endStr, '%d');
        objects{end+1} = objectStruct;
    end
end
fclose(fid);

for a=1:length(actions)
    for o=1:length(actions{a}.objects)
        if(strcmp(actions{a}.objects{o}, 'spoon'))
            actions{a}.objects{o} = 'knife';
        end
        if(strcmp(actions{a}.objects{o}, 'fork'))
            actions{a}.objects{o} = 'knife';
        end
        if(strcmp(actions{a}.objects{o}, 'fork'))
            actions{a}.objects{o} = 'knife';
        end
        if(strcmp(actions{a}.objects{o}, 'forkBox'))
            actions{a}.objects{o} = 'fork';
        end
        if(strcmp(actions{a}.objects{o}, 'plate'))
            actions{a}.objects{o} = 'cupPlateBowl';
        end
        if(strcmp(actions{a}.objects{o}, 'bowl'))
            actions{a}.objects{o} = 'cupPlateBowl';
        end
        if(strcmp(actions{a}.objects{o}, 'cup'))
            actions{a}.objects{o} = 'cupPlateBowl';
        end
        if(strcmp(actions{a}.objects{o}, 'pea'))
            actions{a}.objects{o} = 'beans';
        end
    end
end

