function [ score ] = calcAUCscore( salMap, eyeMap, shufMap, numRandom )
%CALCAUCSCORE Calculate AUC score of a salmap
%   Usage: [score] = calcAUCscore ( salmap, eyemap, shufflemap, numrandom )
%
%   score     : an array of score of each eye fixation
%   salmap    : saliency map. will be resized nearest neighbour to eyemap
%   eyemap    : should be a binary map of eye fixation
%   shufflemap: other image's eye fixation, if undefined will give all
%               white (all white/ones will be random auc instead)
%   numrandom : number of random points sampled from shufflemap
%               default: 100

if nargin < 3
    shufMap = true(size(eyeMap));
end

if nargin < 4
    numRandom = 100;
end

if isempty(shufMap) || max(max(shufMap)) == 0 % its empty or no fixation at all
    shufMap = true(size(eyeMap));
end

%%% Resize and normalize saliency map
salMap = double(imresize(salMap,size(eyeMap),'nearest'));
salMap = salMap - min(min(salMap));
salMap = salMap / max(max(salMap));

%%% Pick saliency value at each eye fixation along with [numrandom] random points
[X Y] = find(eyeMap > 0);
[XRest YRest] = find(shufMap > 0);
localHum = nan(length(X),1);
localRan = nan(length(X),numRandom);
for k=1:length(X)
    localHum(k,1) = salMap(X(k),Y(k));
    for kk=1:numRandom
        r = randi([1 length(XRest)],1);
        localRan(k,kk) = salMap(XRest(r),YRest(r));
    end
end

%%% Calculate AUC score for each eyefix and randpoints
ac = nan(1,size(localRan,2));
R  = cell(1,size(localRan,2));
for ii = 1:size(localRan,2)
    [ac(ii), R{ii}] = nips14.auc(localHum, localRan(:, ii));
end
score = ac;

end

