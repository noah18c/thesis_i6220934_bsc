function predictions = ar_cpd_col(training_series, num_predict, ar_order, cp_order)
    addpath('./tensorlab/');
    % Function to predict future points in a time series using 
    % Canonical Polyadic Decomposition (CPD) and 
    % Autoregressive (AR) modeling. AR is trained on third dimension of the
    % tensor and predicts next point in that dimension, then outerproduct
    % of a, b, and predicted points compute the next num_predict points.
    %
    % Inputs:
    %   training_series - the input time series data for training
    %   num_predict - the number of future points to predict
    %   ar_order - the order of the AR model
    %   cp_order - the number of components for CPD
    %   plotCompare - (optional) boolean flag to plot comparison of results; false by default
    %
    % Outputs:
    %   predictions - the predicted future points

    if nargin < 5
        plotCompare = false;
    end

    % Decomposition using Hankel Tensor (CPD)
    dim_approx = floor(length(training_series)/3);
    
    H3D = hankelize(training_series, 'Sizes', [dim_approx, dim_approx]);
    %disp("Dimension of hankel: "+size(H3D));

    % Compute CPD
    R = cp_order; % Number of components
    [fac,~] = cpd(H3D, R);

    % Initialize all the dimensions of the low-rank tensor approximation
    %a = zeros(size(fac{1},1),1);
    %b = zeros(size(fac{2},1),1);
    %c = zeros(size(fac{3},1),1);

    comp_pred = zeros(num_predict,R);

    % Add all the factors of the components together
    for tcomp = 1:R
        %{

        a = a + fac{1}(:,tcomp);
        b = b + fac{2}(:,tcomp);
        c = c + fac{3}(:,tcomp);
        %}

        a = fac{1}(:,tcomp);
        b = fac{2}(:,tcomp);
        c = fac{3}(:,tcomp);

        model_tcomp = ar(c,ar_order);
        c_predict = forecast(model_tcomp, c, num_predict);

         % Compute the outer product of a, b, and the predicted c values
        final_comp_predict = zeros(num_predict, 1);
        for i = 1:num_predict
            final_comp_predict(i) = a(end) * b(end) * c_predict(i);
        end

        comp_pred(:,tcomp) = final_comp_predict;
    end

    predictions = sum(comp_pred,2);

    %{
    % Predict next point only on c dimension (since this is the direction
    % we want to go in because of the way hankel matrices work)
    model_tcomp = ar(c,ar_order);
    c_predict = forecast(model_tcomp, c, num_predict);
  
    % Compute the outer product of a, b, and the predicted c values
    predictions = zeros(num_predict, 1);
    for i = 1:num_predict
        predictions(i) = a(end) * b(end) * c_predict(i);
    end
    %}
end