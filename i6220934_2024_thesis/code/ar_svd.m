function predictions = ar_svd(training_series, num_predict, ar_order, svd_order ,plotCompare)
    % Function to predict future points in a time series using 
    % Singular Value Decomposition (SVD) and Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   svd_order - the number of components to retain in SVD
    %   plotCompare - (optional) boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points
    if nargin < 5
        plotCompare = false;
    end

    L = floor(length(training_series)/ 2)+1;

    H2D = hankel(training_series(1:L),training_series(L:end));
    [U, S, V] = svd(H2D, 'econ');    

    num_components = svd_order;
    pred_components = zeros(num_predict,num_components);

    if plotCompare
        figure;
        title("Original vs Serialized components")
        plot(training_series);
        hold on;
    end
    for comp = 1:num_components
        %tensor for holding all components (which are matrices)
        matrix_component = U(:,comp)*S(comp,comp)*V(:,comp)';

        % Convert component matrix back to time series by averaging along anti-diagonals
        component_series = serialize2D(matrix_component);

        if plotCompare
            plot(component_series);
            legend({"Original"});
            xlabel("Time");
            ylabel("Value");
            hold off;
        end
        
        model_comp = ar(component_series, ar_order);
        pred_components(:,comp) = forecast(model_comp, component_series, num_predict);
    end


    predictions = sum(pred_components,2);


    if plotCompare
        figure;
        title("Original vs Low-rank approx")
        plot(training_series);
        hold on;
        plot(serial3d);
        legend({"Original", "Low-rank approx"});
        xlabel("Time");
        ylabel("Value");
        hold off;
    end
end