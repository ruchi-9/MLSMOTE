function [train_data, train_target]=MLSMOTE_FUN(X,L,Y,k)
%Description
%
%       LIFT takes,
%				 X  -An NxD array, the ith instance of training instance is stored in X(i,:)
%				 Y  - An NxL array, if the ith training instance belongs to the jth class, then Y(i,j) equals +1, otherwise Y(i,j) equals 0
%				L - A label index for which perform the operation
%  				k - no of nearest neighbour ,By Default 5
%		and returns,
%				train_data - The addition of original X and generated data.
%				train_target- The addition of original Y and generated Label set.

Data=[X,Y];

[r,~]=find(Y(:,L));
minBag=Data(r,:); 

for j=1:size(minBag,1)   
    distance= pdist2(minBag(j,:),minBag); 
    [distance_sort,Idx]=sort(distance,'ascend');  
    %Neighbour set selection
    if size(Idx,2)>=k+1
       neighbour_IDX=Idx(:,2:k+1);
        neighbour=full(minBag(neighbour_IDX,:));
        refNeigh=neighbour(randi(length(neighbour_IDX)),:);
    else
        neighbour_IDX=Idx;
        neighbour=full(minBag(Idx,:));
        refNeigh=neighbour(randi(length(neighbour_IDX)),:);
    end
    
    synthSmpl_feature=zeros(1,size(X,2));
    %feature set assignment
    diff=refNeigh(1,1:size(X,2))-minBag(j,1:size(X,2));
    offset=diff*rand(1);
    value=minBag(j,1:size(X,2))+offset;
    synthSmpl_feature(1,1:size(X,2))=value;
    %label set assignment
    lblCounts=minBag(j,size(X,2)+1:size(minBag,2));
    lblCounts=lblCounts+sum(neighbour(:,size(X,2)+1:size(minBag,2)),1);
    labels=lblCounts>((k+1)/2);
    synthSmpl_label=labels;
    synthSmpl=[synthSmpl_feature,synthSmpl_label];
    
    Data=[Data;synthSmpl];
end
train_data=Data(:,1:size(X,2));
train_target=Data(:,(size(X,2)+1:end));
end