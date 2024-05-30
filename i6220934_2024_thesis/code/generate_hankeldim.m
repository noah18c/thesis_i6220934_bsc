function dimensions = generate_hankeldim(data, dim_range)
    n = length(data);
    dims = length(dim_range);

    dimensions = [];
    i = 1;
    while i<dims && dim_range(i)<n
        f1 = dim_range(i);
        j = 1;
        while j<dims && dim_range(j)<n
            f2 = dim_range(j);
           
            f3 = n-f1-f2+2;
            if(f3>0)
                dimensions(end+1,:) = [f1,f2,f3];
            else
                break;
            end
            
            j=j+1;
        end
        i = i+1;
    end
    


end