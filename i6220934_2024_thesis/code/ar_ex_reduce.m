function [best_reduction, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_reduce(signal_params, max_signals, num_experiments, LM_params,optimal_order, reduction_range)
    addpath('./tensorlab/');

    p = inputParser;

    
    % Max signals parameter
    max_signals_param = size(signal_params, 1);  % You can adjust this as needed

    noise_option = 1;
    num_predict = 1;
    
    % 5 metrics, single model, number of different parameter setups
    all_errors_gt1_mean = zeros(5, 2, max_signals_param, length(reduction_range));
    all_errors_gt2_mean = zeros(5, 2, max_signals_param, length(reduction_range));
    
    for sim_param = 1:max_signals_param
        % Iterate over different reduction values
        for red_idx = 1:length(reduction_range)
            fprintf("Testing reduction value: %.1f\n", reduction_range(red_idx));
            reduction = reduction_range(red_idx);

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
                noisy_series = time_series + noise_option * (randn(N, 1) * 0.4);
                ground_truths2 = noisy_series(end - num_predict + 1:end);
                
                ground_truths = {ground_truths1, ground_truths2};
                training_series = noisy_series(1:end - num_predict);   
                    
                % Prepare storage for predictions
                predictions_mlsvd = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
                rel_error_mlsvd = zeros(num_predict, num_experiments, 2);
                norm_error_mlsvd = zeros(num_predict, num_experiments, 2);
                
                % Perform the experiment
                for experiment = 1:num_experiments 
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    % Generate predictions using ar_mlsvd with current reduction value
                    try
                        predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, reduction, 'L', LM_params(1), 'M', LM_params(2));
                    catch
                        predictions_mlsvd(:, experiment) = NaN;
                    end


                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations
                        rel_error_mlsvd(:, experiment, gt) = abs(predictions_mlsvd(:, experiment) - ground_truths{gt}) ./ abs(ground_truths{gt});
                        norm_error_mlsvd(:, experiment, gt) = norm(predictions_mlsvd(:, experiment), 2) ./ norm(ground_truths{gt}, 2);
                    end
                end   
                       
                % Calculate statistics over number of experiments done
                for gt = 1:2
                    errors_mlsvd = [mean(predictions_mlsvd, 2), std(predictions_mlsvd, 0, 2), rmse(predictions_mlsvd, ground_truths{gt}), mean(rel_error_mlsvd(:, :, gt), 2), mean(norm_error_mlsvd(:, :, gt), 2)]';
                    
                    if gt == 1
                        actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_mlsvd];
                    else
                        actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_mlsvd];
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

    % Select best reduction based on average errors for MLSVD
    % Assuming lower RMSE means better performance
    best_reduction_idx = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_reduction_idx(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 1, sim_param, :)));
    end

    best_reduction = reduction_range(best_reduction_idx);
end