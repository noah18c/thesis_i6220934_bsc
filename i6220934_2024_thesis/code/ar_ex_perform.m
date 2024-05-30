function [all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_perform(signal_params, max_signals, num_experiments, L, LM_params)
    addpath('./tensorlab/');

     % parameters for period, amplitude, sampling frequency (Hz), and interval to be tested
    % example signal parameter setup (needs to have exactly 4 columns)
    %{
    signal_params = [
        1, 1, 1, 100;
        10, 1, 1, 100;
        1, 100, 1, 100;
        10, 100, 1, 100;
        1, 1, 1, 200;
        10, 1, 1, 200;
        1, 100, 1, 200;
        10, 100, 1, 200
    ];
    %}

    % Add default values for optional parameters
    defaultOptimalOrder = 10;
    defaultNumComponents = 1;
    defaultReduction = 0.2;
    defaultEmbedding = 1;
    defaultMethod = @mean;
    
    % Create an input parser
    p = inputParser;
    
    % Add required parameters
    addRequired(p, 'signal_params');
    addRequired(p, 'max_signals');
    addRequired(p, 'num_experiments');
    
    % Add optional name-value pair parameters
    addParameter(p, 'optimal_order', defaultOptimalOrder);
    addParameter(p, 'num_components', defaultNumComponents);
    addParameter(p, 'reduction', defaultReduction);
    addParameter(p, 'embedding', defaultEmbedding);
    addParameter(p, 'method', defaultMethod);
    
    % Parse the inputs
    parse(p, signal_params, max_signals, num_experiments, varargin{:});
    
    % Assign parsed values to variables
    optimal_order = p.Results.optimal_order;
    num_components = p.Results.num_components;
    reduction = p.Results.reduction;
    embedding = p.Results.embedding;
    method = p.Results.method;
    
    max_signals_param = size(signal_params, 1);

    % Total number of simulations:
    % max_signals_param * max_signals * gt * num_experiments
    num_predict = 1;
    noise_option = 1;
    
    % 6 metrics (including real-time duration), 8 models (including actual), number of different parameter setups
    all_errors_gt1_mean = zeros(6, 8, max_signals_param);
    all_errors_gt2_mean = zeros(6, 8, max_signals_param);
    
    for sim_param = 1:max_signals_param
        
        % Number of metrics by number of simulations of different signals for
        % different simulation parameters
        all_errors_gt1 = zeros(6, 8, max_signals);
        all_errors_gt2 = zeros(6, 8, max_signals);
    
        for sim = 1:max_signals
            disp("Parameter simulation " + sim_param + "/" + max_signals_param);
            disp("Generated signal " + sim + "/" + max_signals);
            
            time_series = rsignal(signal_params(sim_param, 1), signal_params(sim_param, 2), signal_params(sim_param, 3), signal_params(sim_param, 4));

            N = length(time_series); % Number of sampling points in the time series
            
            % Ground truths for final point
            ground_truths1 = time_series(end - num_predict + 1:end); 
            noisy_series = time_series + noise_option * (randn(N, 1) * 0.4);
            ground_truths2 = noisy_series(end - num_predict + 1:end);
            
            ground_truths = {ground_truths1, ground_truths2};
            training_series = noisy_series(1:end - num_predict);   
                
            % Prepare storage for predictions
            predictions_AR = zeros(num_predict, num_experiments);
            predictions_SVD = zeros(num_predict, num_experiments);
            predictions_cpd_s = zeros(num_predict, num_experiments);
            predictions_cpd_f = zeros(num_predict, num_experiments);
            predictions_cpd_cols = zeros(num_predict, num_experiments);
            predictions_cpd_colf = zeros(num_predict, num_experiments);
            predictions_mlsvd = zeros(num_predict, num_experiments);
            
            % Prepare storage for errors
            rel_error_AR = zeros(num_predict, num_experiments, 2);
            norm_error_AR = zeros(num_predict, num_experiments, 2);
            rel_error_SVD = zeros(num_predict, num_experiments, 2);
            norm_error_SVD = zeros(num_predict, num_experiments, 2);
            rel_error_cpd_s = zeros(num_predict, num_experiments, 2);
            norm_error_cpd_s = zeros(num_predict, num_experiments, 2);
            rel_error_cpd_f = zeros(num_predict, num_experiments, 2);
            norm_error_cpd_f = zeros(num_predict, num_experiments, 2);
            rel_error_cpd_cols = zeros(num_predict, num_experiments, 2);
            norm_error_cpd_cols = zeros(num_predict, num_experiments, 2);
            rel_error_cpd_colf = zeros(num_predict, num_experiments, 2);
            norm_error_cpd_colf = zeros(num_predict, num_experiments, 2);
            rel_error_mlsvd = zeros(num_predict, num_experiments, 2);
            norm_error_mlsvd = zeros(num_predict, num_experiments, 2);

            % Prepare storage for real-time duration
            duration_AR = zeros(num_experiments, 1);
            duration_SVD = zeros(num_experiments, 1);
            duration_cpd_s = zeros(num_experiments, 1);
            duration_cpd_f = zeros(num_experiments, 1);
            duration_cpd_cols = zeros(num_experiments, 1);
            duration_cpd_colf = zeros(num_experiments, 1);
            duration_mlsvd = zeros(num_experiments, 1);

            % Perform the experiment
            for experiment = 1:num_experiments 
                if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                    disp("iter " + experiment);
                elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                    disp("iter " + experiment);
                end

                % Generate predictions once
                tic;
                model_AR = ar(training_series, optimal_order);  % AR(optimal_order) model
                predictions_AR(:, experiment) = forecast(model_AR, training_series, num_predict);
                duration_AR(experiment) = toc;
                
                tic;
                predictions_SVD(:, experiment) = ar_svd(training_series, num_predict, optimal_order, num_components, 'L', L, 'embedding', embedding, 'Method', method);
                duration_SVD(experiment) = toc;

                tic;
                predictions_cpd_s(:, experiment) = ar_cpd_s(training_series, num_predict, optimal_order, num_components,'L', LM_params(1), 'M', LM_params(2), 'embedding', embedding, 'Method', method);
                duration_cpd_s(experiment) = toc;

                tic;
                predictions_cpd_f(:, experiment) = ar_cpd_f(training_series, num_predict, optimal_order, num_components, 'L', LM_params(1), 'M', LM_params(2),'embedding', embedding, 'Method', method);
                duration_cpd_f(experiment) = toc;
                
                tic;
                predictions_cpd_cols(:, experiment) = ar_cpd_cols(training_series, num_predict, optimal_order, num_components, 'L', LM_params(1), 'M', LM_params(2),'embedding', embedding, 'Method', method);
                duration_cpd_cols(experiment) = toc;

                tic;
                predictions_cpd_colf(:, experiment) = ar_cpd_colf(training_series, num_predict, 10, num_components,'L', LM_params(1), 'M', LM_params(2), 'embedding', embedding, 'Method', method);
                duration_cpd_colf(experiment) = toc;

                tic;
                predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, reduction,'L', LM_params(1), 'M', LM_params(2), 'embedding', embedding, 'Method', method);
                duration_mlsvd(experiment) = toc;

                % Calculate errors for both ground truths
                for gt = 1:2
                    % Error calculations
                    rel_error_AR(:, experiment, gt) = abs(predictions_AR(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_AR(:, experiment, gt) = norm(predictions_AR(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                    rel_error_SVD(:, experiment, gt) = abs(predictions_SVD(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_SVD(:, experiment, gt) = norm(predictions_SVD(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                    rel_error_cpd_s(:, experiment, gt) = abs(predictions_cpd_s(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_cpd_s(:, experiment, gt) = norm(predictions_cpd_s(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                    rel_error_cpd_f(:, experiment, gt) = abs(predictions_cpd_f(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_cpd_f(:, experiment, gt) = norm(predictions_cpd_f(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    
                    rel_error_cpd_cols(:, experiment, gt) = abs(predictions_cpd_cols(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_cpd_cols(:, experiment, gt) = norm(predictions_cpd_cols(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                    rel_error_cpd_colf(:, experiment, gt) = abs(predictions_cpd_colf(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_cpd_colf(:, experiment, gt) = norm(predictions_cpd_colf(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    
                    rel_error_mlsvd(:, experiment, gt) = abs(predictions_mlsvd(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                    norm_error_mlsvd(:, experiment, gt) = norm(predictions_mlsvd(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                end
            end   
                   
            % Calculate statistics over number of experiments done
            for gt = 1:2
                errors_AR = [mean(predictions_AR, 2), std(predictions_AR, 0, 2), rmse(predictions_AR, ground_truths{gt}, 2), mean(rel_error_AR(:, :, gt), 2), mean(norm_error_AR(:, :, gt), 2), mean(duration_AR)]';
                errors_SVD = [mean(predictions_SVD, 2), std(predictions_SVD, 0, 2), rmse(predictions_SVD, ground_truths{gt}, 2), mean(rel_error_SVD(:, :, gt), 2), mean(norm_error_SVD(:, :, gt), 2), mean(duration_SVD)]';
                errors_cpd_s = [mean(predictions_cpd_s, 2), std(predictions_cpd_s, 0, 2), rmse(predictions_cpd_s, ground_truths{gt}, 2), mean(rel_error_cpd_s(:, :, gt), 2), mean(norm_error_cpd_s(:, :, gt), 2), mean(duration_cpd_s)]';
                errors_cpd_f = [mean(predictions_cpd_f, 2), std(predictions_cpd_f, 0, 2), rmse(predictions_cpd_f, ground_truths{gt}, 2), mean(rel_error_cpd_f(:, :, gt), 2), mean(norm_error_cpd_f(:, :, gt), 2), mean(duration_cpd_f)]';
                errors_cpd_cols = [mean(predictions_cpd_cols, 2), std(predictions_cpd_cols, 0, 2), rmse(predictions_cpd_cols, ground_truths{gt}, 2), mean(rel_error_cpd_cols(:, :, gt), 2), mean(norm_error_cpd_cols(:, :, gt), 2), mean(duration_cpd_cols)]';
                errors_cpd_colf = [mean(predictions_cpd_colf, 2), std(predictions_cpd_colf, 0, 2), rmse(predictions_cpd_colf, ground_truths{gt}, 2), mean(rel_error_cpd_colf(:, :, gt), 2), mean(norm_error_cpd_colf(:, :, gt), 2), mean(duration_cpd_colf)]';
                errors_mlsvd = [mean(predictions_mlsvd, 2), std(predictions_mlsvd, 0, 2), rmse(predictions_mlsvd, ground_truths{gt}, 2), mean(rel_error_mlsvd(:, :, gt), 2), mean(norm_error_mlsvd(:, :, gt), 2), mean(duration_mlsvd)]';
                
                % Depending on ground truth, set stats to either errors_concat1 or 2
                if gt == 1
                    actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0, 0]';
                    errors_concat1 = [actual_stats, errors_AR, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                else
                    actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0, 0]';
                    errors_concat2 = [actual_stats, errors_AR, errors_SVD, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                end
            end
    
            % Add matrix of error calculations as a slice to 3D matrix
            all_errors_gt1(:, :, sim) = errors_concat1;
            all_errors_gt2(:, :, sim) = errors_concat2;
        end
        
        % Below are the average evaluation scores (mean of matrix slices)
        % rows: mean; sd; RMSE; MRE; MRSE; duration
        % columns: AR, SVD, CPD_S, CPD_F, CPD_Cols, CPD_ColF, MLSVD
        % slices: results per signal parameter set
        all_errors_gt1_mean(:, :, sim_param) = mean(all_errors_gt1, 3);
        all_errors_gt2_mean(:, :, sim_param) = mean(all_errors_gt2, 3);
    
    end
    disp("Done!");
end
