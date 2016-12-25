function [ X_norm,avg,sigma ] = normalize(X)
%Z-score Normalization
avg = mean(X,1);
sigma = std(X,1);
X_norm  = (X - repmat(avg,size(X,1),1)) ./  repmat(sigma,size(X,1),1);
end
