function predictions = ar_mlsvd(training_series, num_predict, ar_order, threshold, varargin)
    % Function to predict future points in a time series using 
    % Multilinear Singular Value Decomposition (MLSVD) and 
    % Autoregressive (AR) modeling.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   threshold - the threshold factor to determine the size of the core
    %   tensor (between 0 and 1, lower is more threshold)
    %   num_predict - the number of future points to predict
    %   varargin - optional name-value pair arguments:
    %       'L' - the first dimension of the Hankel tensor
    %       'M' - the second dimension of the Hankel tensor
    %       'plotCompare' - boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points

    % Define default values for optional parameters
    defaultEmbedding = 1; % 1 is Hankel, 2 is segmentation, 3 is decimation
    defaultPlotCompare = false;
    defaultEven = false; % true = remove first value of uneven sequences for embedding 2
    defaultMethod = @mean;

    % Create an input parser
    p = inputParser;

    % Add required parameters
    addRequired(p, 'training_series', @isnumeric);
    addRequired(p, 'num_predict', @isnumeric);
    addRequired(p, 'ar_order', @isnumeric);
    addRequired(p, 'threshold', @(x) isnumeric(x) && x > 0 && x <= 1);

    % Add optional name-value pair parameters
    addParameter(p, 'embedding', defaultEmbedding, @isnumeric);
    addParameter(p, 'L', [], @isnumeric);
    addParameter(p, 'M', [], @isnumeric);
    addParameter(p, 'even', defaultEven, @islogical);
    addParameter(p, 'plotCompare', defaultPlotCompare, @islogical);
    addParameter(p, 'Method', defaultMethod);

    % Parse inputs
    parse(p, training_series, num_predict, ar_order, threshold, varargin{:});

    % Extract values from the input parser
    embedding = p.Results.embedding;
    plotCompare = p.Results.plotCompare;
    evenSequence = p.Results.even;
    method = p.Results.Method;

    % Set default values for L and M based on embedding method
    if embedding == 1
        defaultL = floor(length(training_series) / 3);
        defaultM = floor(length(training_series) / 3);
    elseif embedding == 2 || embedding == 3
        defaultL = floor(length(training_series)^(1/3));
        defaultM = floor(length(training_series)^(1/3));
    else
        error('Invalid embedding value.');
    end

    % Override defaults if L and M are provided
    L = defaultL;
    if ~isempty(p.Results.L)
        L = p.Results.L;
    end
    M = defaultM;
    if ~isempty(p.Results.M)
        M = p.Results.M;
    end

    % Ensure L and M are within valid range
    if L > length(training_series) || L < 1
        error('L must be a positive integer less than or equal to the length of the training series.');
    end
    if M > length(training_series) || M < 1
        error('M must be a positive integer less than or equal to the length of the training series.');
    end

    switch embedding
        case 1
            % Approximate the dimensions so that H3D is almost cubic
            H3D = hankelize(training_series, 'Sizes', [L, M]);
            %disp("Dimension of hankel: "+size(H3D));
        
            [U, C, sv] = mlsvd(H3D);

            R =[1;1;1];
            
            % determine the size of the core based off of the singular values a certain
            % threshold of the singular values compared to the most significant one.
            for dim = 1:3
                for i = 2:length(sv{dim})
                    if sv{dim}(i) > threshold*sv{dim}(1)
                        R(dim) = R(dim) + 1;
                    end
                end
            end
            
            Utrunc{1} = U{1}(:,1:R(1)); % Column space orthogonal base
            Utrunc{2} = U{2}(:,1:R(2)); % Row-space orthogonal base
            Utrunc{3} = U{3}(:,1:R(3)); % Fiber-space orthogonal base
            
            Ctrunc = C(1:R(1),1:R(2),1:R(3)); % truncated core
            
            low_rank_tensor = lmlragen(Utrunc, Ctrunc);
            
            serial3d = dehankelize(low_rank_tensor, 'Dims', 1:3, 'Method', method);   
            
            model = ar(serial3d, ar_order);
            predictions = forecast(model, serial3d, num_predict);
        
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

            % Approximate the dimensions so that S3D is almost cubic
            S3D = segmentize(training_series,3,'Segsize',[L M],'UseAllSamples',true);
            %disp("Dimension of hankel: "+size(H3D));
        
            [U, C, sv] = mlsvd(S3D);

            R =[1;1;1];
            
            % determine the size of the core based off of the singular values a certain
            % threshold of the singular values compared to the most significant one.
            for dim = 1:3
                for i = 2:length(sv{dim})
                    if sv{dim}(i) > threshold*sv{dim}(1)
                        R(dim) = R(dim) + 1;
                    end
                end
            end
            
            Utrunc{1} = U{1}(:,1:R(1)); % Column space orthogonal base
            Utrunc{2} = U{2}(:,1:R(2)); % Row-space orthogonal base
            Utrunc{3} = U{3}(:,1:R(3)); % Fiber-space orthogonal base
            
            Ctrunc = C(1:R(1),1:R(2),1:R(3)); % truncated core
            
            low_rank_tensor = lmlragen(Utrunc, Ctrunc);
            
            serial3d = desegmentize(low_rank_tensor,'Dims',1:3, 'Method', method);
            
            model = ar(serial3d, ar_order);
            predictions = forecast(model, serial3d, num_predict);
        
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

            % Approximate the dimensions so that S3D is almost cubic
            D3D = decimate(training_series,'Nsamples',[L M],'UseAllSamples',true);
            %disp("Dimension of hankel: "+size(H3D));
        
            [U, C, sv] = mlsvd(D3D);

            R =[1;1;1];
            
            % determine the size of the core based off of the singular values a certain
            % threshold of the singular values compared to the most significant one.
            for dim = 1:3
                for i = 2:length(sv{dim})
                    if sv{dim}(i) > threshold*sv{dim}(1)
                        R(dim) = R(dim) + 1;
                    end
                end
            end
            
            Utrunc{1} = U{1}(:,1:R(1)); % Column space orthogonal base
            Utrunc{2} = U{2}(:,1:R(2)); % Row-space orthogonal base
            Utrunc{3} = U{3}(:,1:R(3)); % Fiber-space orthogonal base
            
            Ctrunc = C(1:R(1),1:R(2),1:R(3)); % truncated core
            
            low_rank_tensor = lmlragen(Utrunc, Ctrunc);
            
            serial3d = dedecimate(low_rank_tensor,'Dims',1:3,'Method', method);
            
            model = ar(serial3d, ar_order);
            predictions = forecast(model, serial3d, num_predict);
        
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