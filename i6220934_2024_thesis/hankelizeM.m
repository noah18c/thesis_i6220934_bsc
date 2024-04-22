function H = hankelizeM(matrix, L)
    c = matrix(1,1:L);
    r = matrix(1,L:size(matrix,2));
    H = hankel(c,r);
end