function [thresh, comp] = select_param(perfInd, tr, cr)

    
    thresh = zeros(2,1);
    comp = zeros(4,1);
    j = 1;
    k = 1;
    for i = 2:size(perfInd,2)
        if i == 2 || i == 7
            thresh(j) = tr(perfInd(i));
            j = j + 1;
        else
            comp(k) = cr(perfInd(i));
            k = k + 1;
        end
    end

end