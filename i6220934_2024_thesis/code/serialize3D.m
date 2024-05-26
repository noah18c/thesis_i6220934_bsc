function reconstructed_data = serialize3D(H3D)
    ts_length = size(H3D,1)+size(H3D,2)+size(H3D,3)-2;
    
    reconstructed_data = zeros(ts_length,1);
    counts = zeros(ts_length, 1);
    
    for i = 1:size(H3D,1)
        for j = 1:size(H3D,2)
            for k = 1:size(H3D,3)
                reconstructed_data(i+j+k-2) = reconstructed_data(i+j+k-2) + H3D(i, j, k);
                counts(i+j+k-2) = counts(i+j+k-2) + 1;
            end
        end
    end
    reconstructed_data = reconstructed_data ./ counts;
end

