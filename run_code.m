clear
%% load the dataset
load('emotions.mat');
k = 5; % no of nearest neighbour
feature_set   = [X;Xt];
label_set     = [Y;Yt];
final_feature = [];
final_label   = [];
clear X Xt Y Yt
%% find the minority label
labelWiseIns = sum(label_set,1);
IR_label     = max(labelWiseIns)./(eps+labelWiseIns);
meanir       = sum(IR_label)/size(label_set,2);
minorityL    = IR_label>meanir;
[~,numL] = size(label_set);

for L = 1:numL
    if minorityL(L)
        [train_data, train_target]  =  MLSMOTE_FUN(full(feature_set),L,full(label_set),k);
        final_feature = [final_feature;train_data];
        final_label = [final_label;train_target];
    end
end
% Remove the Duplicate
Data    =   unique([final_feature final_label],'rows');
final_feature   =   Data(:,1:size(final_feature,2));
final_label     =   Data(:,(size(final_feature,2)+1):end);

