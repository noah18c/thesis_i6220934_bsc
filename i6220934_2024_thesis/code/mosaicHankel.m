function Hw = mosaicHankel(u, y, lag)
    L = lag;  % Number of lags (rows in the Hankel matrix)

    Hu = hankel(u(1:L),u(L:end));
    Hy = hankel(y(1:L),y(L:end));
    
    Hw = [Hu; Hy];
end