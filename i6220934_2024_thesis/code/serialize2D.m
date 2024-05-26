function reconstructed_data = serialize2D(H2D)
    ts_length = size(H2D,1)+size(H2D,2)-1;
    
    reconstructed_data = zeros(ts_length,1);
    counts = zeros(ts_length, 1);

    for i = 1:size(H2D,1)
        for j = 1:size(H2D,2)
            reconstructed_data(i+j-1) = reconstructed_data(i+j-1) + H2D(i,j);
            counts(i+j-1) = counts(i+j-1) + 1;
        end
    end
    reconstructed_data = reconstructed_data ./ counts;
end