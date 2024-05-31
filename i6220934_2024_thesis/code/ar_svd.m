function predictions = ar_svd(training_series, num_predict, ar_order, threshold, varargin)
    % Function to predict future points in a time series using 
    % Singular Value Decomposition (SVD) and Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   svd_order - the number of components to retain in SVD
    %   varargin - optional name-value pair arguments:
    %       'L' - the length of the first column of the Hankel matrix
    %       'plotCompare' - boolean flag to plot comparison of results; false by default
    %       'embedding' - the embedding method: 1 for Hankel, 2 for segmentation
    %
    % Outputs:
    %   predictions - the predicted future points

    % Define default values for optional parameters
    defaultPlotCompare = false;
    defaultEmbedding = 1; % 1 is Hankel, 2 is segmentation, 3 is decimation
    defaultMethod = @mean;

    % Create an input parser
    p = inputParser;

    % Add required parameters
    addRequired(p, 'training_series', @isnumeric);
    addRequired(p, 'num_predict', @isnumeric);
    addRequired(p, 'ar_order', @isnumeric);
    addRequired(p, 'threshold', @isnumeric);

    % Add optional name-value pair parameters
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);
    addParameter(p, 'embedding', defaultEmbedding, @isnumeric);
    addParameter(p, 'Method', defaultMethod);
    addParameter(p, 'L', [], @isnumeric);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, threshold, varargin{:});

    % Extract values from the input parser
    plotCompare = p.Results.plotCompare;
    embedding = p.Results.embedding;
    method = p.Results.Method;

    % Set default values for L and M based on embedding method
    if embedding == 1
        defaultL = floor(length(training_series) / 3);
    elseif embedding == 2 || embedding == 3
        defaultL = floor(length(training_series)^(1/3));
    else
        error('Invalid embedding value.');
    end

    % Override defaults if L is provided
    L = defaultL;
    if ~isempty(p.Results.L)
        L = p.Results.L;
    end

    % Ensure L is within valid range
    if L > length(training_series) || L < 1
        error('L must be a positive integer less than or equal to the length of the training series.');
    end

    % Create Hankel matrix or segmented matrix based on embedding method
    switch embedding
        case 1
            H2D = hankel(training_series(1:L), training_series(L:end));
            [U, S, V] = svd(H2D, 'econ');
            
            num_components = 1;
            for i = 2:size(S,1)
                if S(i,i) > threshold*S(1,1)
                        num_components = num_components+1;
                end
            end
        
            pred_components = zeros(num_predict, num_components);

            
        
            if plotCompare
                figure;
                title("Original vs Serialized components")
                plot(training_series);
                hold on;
            end
        
            for comp = 1:num_components
                % Tensor for holding all components (which are matrices)
                matrix_component = U(:,comp) * S(comp, comp) * V(:,comp)';
        
                % Convert component matrix back to time series by averaging along anti-diagonals
                component_series = dehankelize(matrix_component, 'Method', method);
        
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
        
            predictions = sum(pred_components, 2);
        
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
        case 2
            S2D = segmentize(training_series,2,'Segsize',L,'UseAllSamples',true);
            [U, S, V] = svd(S2D, 'econ');    
        
            num_components = 1;
            for i = 2:size(S,1)
                if S(i,i) > threshold*S(1,1)
                        num_components = num_components+1;
                end
            end
        
            pred_components = zeros(num_predict, num_components);
        
            if plotCompare
                figure;
                title("Original vs Serialized components")
                plot(training_series);
                hold on;
            end
        
            for comp = 1:num_components
                % Tensor for holding all components (which are matrices)
                matrix_component = U(:,comp) * S(comp, comp) * V(:,comp)';
        
                % Convert component matrix back to time series by reshaping
                component_series = desegmentize(matrix_component, 'Method', method);
        
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
        
            predictions = sum(pred_components, 2);
        
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
        case 3
           
            D2D = decimate(training_series,'Nsamples',L,'UseAllSamples',true);
            [U, S, V] = svd(D2D, 'econ');    
        
            num_components = 1;
            for i = 2:size(S,1)
                if S(i,i) > threshold*S(1,1)
                        num_components = num_components+1;
                end
            end
        
            pred_components = zeros(num_predict, num_components);
        
            if plotCompare
                figure;
                title("Original vs Serialized components")
                plot(training_series);
                hold on;
            end
        
            for comp = 1:num_components
                % Tensor for holding all components (which are matrices)
                matrix_component = U(:,comp) * S(comp, comp) * V(:,comp)';
        
                % Convert component matrix back to time series by reshaping
                component_series = dedecimate(matrix_component, 'Method', method);
        
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
        
            predictions = sum(pred_components, 2);
        
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
end
