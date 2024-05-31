function [best_thresholds, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_threshold(signal_params, max_signals, num_experiments, LM_params, optimal_order, threshold_range, embedding, varargin)
    addpath('./tensorlab/');

    p = inputParser;

    addRequired(p, 'signal_params');
    addRequired(p, 'max_signals');
    addRequired(p, 'num_experiments');
    addRequired(p, 'LM_params');
    addRequired(p, 'threshold_range');

    addParameter(p, 'noise_param', 0.1);
    addParameter(p, 'embedding', 1);
    addParameter(p, 'optimal_order', 10);

    parse(p, signal_params, max_signals, num_experiments, LM_params, threshold_range, varargin{:});

    % Assign parsed values to variables
    optimal_order = p.Results.optimal_order;
    embedding = p.Results.embedding;
    noise_param = p.Results.noise_param;

    % Max signals parameter
    max_signals_param = size(signal_params, 1);  % You can adjust this as needed

    num_predict = 1;
    
    % 5 metrics, 2 models (svd and mlsvd), number of different parameter setups
    all_errors_gt1_mean = zeros(5, 2, max_signals_param, length(threshold_range));
    all_errors_gt2_mean = zeros(5, 2, max_signals_param, length(threshold_range));
    
    for sim_param = 1:max_signals_param
        % Iterate over different threshold values
        for red_idx = 1:length(threshold_range)
            fprintf("Testing threshold value: %.1f\n", threshold_range(red_idx));
            threshold = threshold_range(red_idx);

            % Number of metrics by number of simulations of different signals for
            % different simulation parameters
            all_errors_gt1 = zeros(5, 2, max_signals);
            all_errors_gt2 = zeros(5, 2, max_signals);
        
            for sim = 1:max_signals
                disp("Parameter simulation " + sim_param + "/" + max_signals_param);
                disp("Generated signal " + sim + "/" + max_signals);
                
                time_series = rsignal(signal_params(sim_param, 1), signal_params(sim_param, 2), signal_params(sim_param, 3), signal_params(sim_param, 4));

                N = length(time_series); % Number of sampling points in the time series

                % Ground truth for final point
                ground_truths1 = time_series(end - num_predict + 1:end); 
                
                % create noise on series proportionate to its range
                rt = range(time_series);
                noise = (1-2.*round(rand(N,1))).*(noise_param*rand(N,1)*rt);
                noisy_series = time_series + noise;

                ground_truths2 = noisy_series(end - num_predict + 1:end);
                
                ground_truths = {ground_truths1, ground_truths2};
                training_series = noisy_series(1:end - num_predict);   
                    
                % Prepare storage for predictions
                predictions_svd = zeros(num_predict, num_experiments);
                predictions_mlsvd = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
                rel_error_svd = zeros(num_predict, num_experiments, 2);
                norm_error_svd = zeros(num_predict, num_experiments, 2);
                rel_error_mlsvd = zeros(num_predict, num_experiments, 2);
                norm_error_mlsvd = zeros(num_predict, num_experiments, 2);
                
                % Perform the experiment
                for experiment = 1:num_experiments 
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    % Generate predictions using ar_svd and ar_mlsvd with current threshold value
                    try
                        predictions_svd(:, experiment) = ar_svd(training_series, num_predict, optimal_order, threshold, 'L', LM_params(1,1), 'M', LM_params(1,1),'embedding', embedding);
                    catch
                        predictions_svd(:, experiment) = NaN;
                    end

                    try
                        predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, threshold, 'L', LM_params(6,1), 'M', LM_params(6,1),'embedding', embedding);
                    catch
                        predictions_mlsvd(:, experiment) = NaN;
                    end

                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations for SVD
                        rel_error_svd(:, experiment, gt) = abs(predictions_svd(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_svd(:, experiment, gt) = norm(predictions_svd(:, experiment), 2) ./ norm(ground_truths{gt}, 2);

                        % Error calculations for MLSVD
                        rel_error_mlsvd(:, experiment, gt) = abs(predictions_mlsvd(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_mlsvd(:, experiment, gt) = norm(predictions_mlsvd(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    end
                end   
                       
                % Calculate statistics over number of experiments done
                for gt = 1:2
                    errors_svd = [mean(predictions_svd, 2), std(predictions_svd, 0, 2), rmse(predictions_svd, ground_truths{gt}), mean(rel_error_svd(:, :, gt), 2), mean(norm_error_svd(:, :, gt), 2)]';
                    errors_mlsvd = [mean(predictions_mlsvd, 2), std(predictions_mlsvd, 0, 2), rmse(predictions_mlsvd, ground_truths{gt}), mean(rel_error_mlsvd(:, :, gt), 2), mean(norm_error_mlsvd(:, :, gt), 2)]';
                    
                    if gt == 1
                        errors_concat1 = [errors_svd, errors_mlsvd];
                    else
                        errors_concat2 = [errors_svd, errors_mlsvd];
                    end
                end

                % Add matrix of error calculations as a slice to 3D matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % Average evaluation scores over all signals
            all_errors_gt1_mean(:, :, sim_param, red_idx) = mean(all_errors_gt1, 3);
            all_errors_gt2_mean(:, :, sim_param, red_idx) = mean(all_errors_gt2, 3);
        end
    end
    disp("Done!");

    % Select best threshold based on average errors for SVD and MLSVD
    % Assuming lower RMSE means better performance
    best_threshold_svd_idx = zeros(1, max_signals_param);
    best_threshold_mlsvd_idx = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_threshold_svd_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 1, sim_param, :)));
        [~, best_threshold_mlsvd_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 2, sim_param, :)));
    end

    best_thresholds = struct('svd', threshold_range(best_threshold_svd_idx), 'mlsvd', threshold_range(best_threshold_mlsvd_idx));
end
