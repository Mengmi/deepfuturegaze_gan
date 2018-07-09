function [ score ] = calcNSSscore( salMap, eyeMap )
%calcNSSscore Calculate NSS score of a salmap
%   Usage: [score] = calcNSSscore ( salmap, eyemap )
%
%   score     : an array of score of each eye fixation
%   salmap    : saliency map. will be resized nearest neighbour to eyemap
%   eyemap    : should be a binary map of eye fixation


%%% Resize and normalize saliency map
salMap = double(imresize(salMap,size(eyeMap),'nearest'));
mapMean = mean2(salMap); mapStd = std2(salMap);
salMap = (salMap - mapMean) / mapStd; % Normalized map

%%% NSS calculation
[X Y] = find(eyeMap > 0);
NSSVector = zeros(1,size(X,1));
for p=1:size(X,1)
    NSSVector(p) = salMap(X(p),Y(p));
end
score = NSSVector;

end

