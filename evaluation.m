% input has [image, location, probability]
load CityCentre_GroundTruth
%input = load('-ascii', '_fabmap_ox_CityCentre_cbCityCentre_c11000_bay_s20.txt');
%input = load('-ascii', '_fabmap_ox_CityCentre_cbCityCentre_c11000_clt_s20.txt');
input = load('-ascii', '_fabmap_onlinerun_statistics.txt');
% check performance
%% build image-image correspondence matrix
numImages = size(input,1);
ImageCorr = zeros(numImages,numImages);
for i=1:numImages
	for j=1:numImages
		if (input(i,2) == input(j,2))
			% both images were assigned to same location
			ImageCorr(i, j) = 1;
		end
	end
end
ImageCorr = ImageCorr - eye(numImages,numImages);

%% compare with ground truth matrix

A = truth-ImageCorr;

truePos = 0;
falsePos = 0;
falseNeg = 0;
trueNeg = 0;
for i=1:numImages
    for j=1:numImages
        if (A(i,j)==-1)
            falsePos = falsePos+1;
        end
        if (A(i,j)==1)
            falseNeg = falseNeg+1;
        end
        if (truth(i,j)==1 && ImageCorr(i,j)==1)
            truePos = truePos+1;
        end
        if (truth(i,j)==0 && ImageCorr(i,j)==0)
            trueNeg = trueNeg+1;
        end
    end
end
truePos
falsePos
falseNeg
trueNeg

% precision = TP / (TP + FP)
% recall = FP / (FP + TN)
precision = truePos/(truePos+falsePos)
recall = truePos/(falseNeg+truePos)   % here recall is defined as found matches/possible matches                         %falsePos/(falsePos+trueNeg)
