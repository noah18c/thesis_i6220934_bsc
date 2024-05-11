function H = hankelizeM(matrix, lags)
    c = matrix(1,1:lags);
    r = matrix(1,lags:size(matrix,2));
    H = hankel(c,r);
end