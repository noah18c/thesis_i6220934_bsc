function [best_L, all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex_tensor(signal_params, LM_range, max_signals, num_experiments, varargin)
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
    defaultReduction = 0.001;
    defaultEmbedding = 1;
    defaultMethod = @mean;
    
    % Create an input parser
    p = inputParser;
    
    % Add required parameters
    addRequired(p, 'signal_params');
    addRequired(p, 'LM_range');
    addRequired(p, 'max_signals');
    addRequired(p, 'num_experiments');
    
    % Add optional name-value pair parameters
    addParameter(p, 'optimal_order', defaultOptimalOrder);
    addParameter(p, 'num_components', defaultNumComponents);
    addParameter(p, 'reduction', defaultReduction);
    addParameter(p, 'embedding', defaultEmbedding);
    addParameter(p, 'method', defaultMethod);
    
    % Parse the inputs
    parse(p, signal_params, LM_range, max_signals, num_experiments, varargin{:});
    
    % Assign parsed values to variables
    optimal_order = p.Results.optimal_order;
    num_components = p.Results.num_components;
    reduction = p.Results.reduction;
    embedding = p.Results.embedding;
    method = p.Results.method;
    
    % Max signals parameter
    max_signals_param = size(signal_params, 1);  % You can adjust this as needed

    num_predict = 1;
    
    % 5 metrics, 5 models (including actual), number of different parameter setups
    all_errors_gt1_mean = zeros(5, 6, max_signals_param, size(LM_range,1));
    all_errors_gt2_mean = zeros(5, 6, max_signals_param, size(LM_range,1));
    
    for sim_param = 1:max_signals_param
        disp("Parameter simulation " + sim_param + "/" + max_signals_param);
        % Iterate over different values of L
        for dim_test = 1:size(LM_range,1)
            disp("Testing dimensions values: "+ LM_range(dim_test,1)+"x"+LM_range(dim_test,2)+"x"+LM_range(dim_test,3));
            L = LM_range(dim_test,1);
            M = LM_range(dim_test,2);

            % Number of metrics by number of simulations of different signals for
            % different simulation parameters
            all_errors_gt1 = zeros(5, 6, max_signals);
            all_errors_gt2 = zeros(5, 6, max_signals);
        
            for sim = 1:max_signals
                disp("Generated signal " + sim + "/" + max_signals);

                %this parameter exist such that the learning part is always
                %of constant length no matter how many predictions you
                %want.
                lookout = num_predict - 1;
                time_series = rsignal(signal_params(sim_param, 1), signal_params(sim_param, 2), signal_params(sim_param, 3), signal_params(sim_param, 4)+lookout);

                N = length(time_series); % Number of sampling points in the time series

                % Ground truth for final point
                ground_truths1 = time_series(end - num_predict + 1:end); 

                % create noise on series proportionate to its range
                noise_param = signal_params(sim_param,5);
                noisy_series = noisify(noise_param,N,time_series);

                ground_truths2 = noisy_series(end - num_predict + 1:end);
                
                ground_truths = {ground_truths1, ground_truths2};
                training_series = noisy_series(1:end - num_predict);   
                    
                % Prepare storage for predictions
                predictions_cpd_s = zeros(num_predict, num_experiments);
                predictions_cpd_f = zeros(num_predict, num_experiments);
                predictions_cpd_cols = zeros(num_predict, num_experiments);
                predictions_cpd_colf = zeros(num_predict, num_experiments);
                predictions_mlsvd = zeros(num_predict, num_experiments);
                
                % Prepare storage for errors
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
                
                % Perform the experiment
                for experiment = 1:num_experiments                        
                    if mod(experiment, round(num_experiments / 4)) == 0 && mod(experiment, 2) == 0
                        disp("iter " + experiment);
                    elseif mod(experiment, round(num_experiments / 4)) == 0 || experiment == num_experiments
                        disp("iter " + experiment);
                    end

                    % In case anything goes wrong we use NaN to indicate
                    % that this is not a desirable dimension for the tensor

                    try
                        predictions_cpd_s(:, experiment) = ar_cpd_s(training_series, num_predict, optimal_order, num_components, 'L', L, 'M', M, 'embedding', embedding, 'Method', method);
                    catch
                        predictions_cpd_s(:, experiment) = NaN;
                    end
                    try
                        predictions_cpd_f(:, experiment) = ar_cpd_f(training_series, num_predict, optimal_order, num_components, 'L', L, 'M', M, 'embedding', embedding, 'Method', method);
                    catch
                        predictions_cpd_f(:, experiment) = NaN;
                    end

                    % When dimensions are too small in the required dimension it generates an error
                    % because the AR will have too little data to be
                    % trained on.
                    try
                        predictions_cpd_cols(:, experiment) = ar_cpd_cols(training_series, num_predict, 2, num_components, 'L', L, 'M', M, 'embedding', embedding, 'Method', method);
                    catch
                        predictions_cpd_cols(:, experiment) = NaN;
                    end
                    try
                        predictions_cpd_colf(:, experiment) = ar_cpd_colf(training_series, num_predict, 2, num_components, 'L', L, 'M', M, 'embedding', embedding, 'Method', method);
                    catch
                        predictions_cpd_colf(:, experiment) = NaN;
                    end
                    try
                        predictions_mlsvd(:, experiment) = ar_mlsvd(training_series, num_predict, optimal_order, reduction, 'L', L, 'M', M, 'embedding', embedding, 'Method', method);
                    catch
                        predictions_mlsvd(:, experiment) = NaN;
                    end

                    % Calculate errors for both ground truths
                    for gt = 1:2
                        % Error calculations
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
                    errors_cpd_s = [mean(predictions_cpd_s, 2), std(predictions_cpd_s, 0, 2), rmse(predictions_cpd_s, ground_truths{gt}, 2), mean(rel_error_cpd_s(:, :, gt), 2), mean(norm_error_cpd_s(:, :, gt), 2)]';
                    errors_cpd_f = [mean(predictions_cpd_f, 2), std(predictions_cpd_f, 0, 2), rmse(predictions_cpd_f, ground_truths{gt}, 2), mean(rel_error_cpd_f(:, :, gt), 2), mean(norm_error_cpd_f(:, :, gt), 2)]';
                    errors_cpd_cols = [mean(predictions_cpd_cols, 2), std(predictions_cpd_cols, 0, 2), rmse(predictions_cpd_cols, ground_truths{gt}, 2), mean(rel_error_cpd_cols(:, :, gt), 2), mean(norm_error_cpd_cols(:, :, gt), 2)]';
                    errors_cpd_colf = [mean(predictions_cpd_colf, 2), std(predictions_cpd_colf, 0, 2), rmse(predictions_cpd_colf, ground_truths{gt}, 2), mean(rel_error_cpd_colf(:, :, gt), 2), mean(norm_error_cpd_colf(:, :, gt), 2)]';
                    errors_mlsvd = [mean(predictions_mlsvd, 2), std(predictions_mlsvd, 0, 2), rmse(predictions_mlsvd, ground_truths{gt}, 2), mean(rel_error_mlsvd(:, :, gt), 2), mean(norm_error_mlsvd(:, :, gt), 2)]';
                    
                    if gt == 1
                        actual_stats = [mean(ground_truths{gt}, 2), 0, 0, 0, 0]';
                        errors_concat1 = [actual_stats, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                    else
                        actual_stats = [mean(ground_truths{gt}, 2), std(ground_truths{gt}, 0, 2), 0, 0, 0]';
                        errors_concat2 = [actual_stats, errors_cpd_s, errors_cpd_f, errors_cpd_cols, errors_cpd_colf, errors_mlsvd];
                    end
                end

                % Add matrix of error calculations as a slice to 3d matrix
                all_errors_gt1(:, :, sim) = errors_concat1;
                all_errors_gt2(:, :, sim) = errors_concat2;
            end
            
            % Below are the average evaluation scores (mean of matrix slices)
            % rows: mean; sd; RMSE; MRE; MRSE
            % columns: cpd_s, cpd_f, cpd_cols, cpd_colf, mlsvd
            % slices: results per signal parameter set
            all_errors_gt1_mean(:, :, sim_param, dim_test) = mean(all_errors_gt1, 3);
            all_errors_gt2_mean(:, :, sim_param, dim_test) = mean(all_errors_gt2, 3);
        end
    end
    disp("Done!");

    % Select best L based on average errors for CPD-S, CPD-F, CPD-cols, CPD-ColF, and MLSVD
    % Here, assuming lower RMSE means better performance.
    % Rows: mean; sd; RMSE; MRE; MRSE
    % Columns: cpd_s, cpd_f, cpd_cols, cpd_colf, mlsvd
    best_cpd_s_dim_test = zeros(1, max_signals_param);
    best_cpd_f_dim_test = zeros(1, max_signals_param);
    best_cpd_cols_dim_test = zeros(1, max_signals_param);
    best_cpd_colf_dim_test = zeros(1, max_signals_param);
    best_mlsvd_dim_test = zeros(1, max_signals_param);
    
    for sim_param = 1:max_signals_param
        [~, best_cpd_s_dim_test(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 1, sim_param, :)));
        [~, best_cpd_f_dim_test(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 2, sim_param, :)));
        [~, best_cpd_cols_dim_test(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 3, sim_param, :)));
        [~, best_cpd_colf_dim_test(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 4, sim_param, :)));
        [~, best_mlsvd_dim_test(sim_param)] = min(squeeze(all_errors_gt1_mean(3, 5, sim_param, :)));
    end

    best_L = struct('cpd_s', LM_range(best_cpd_s_dim_test,1:2), 'cpd_f', LM_range(best_cpd_f_dim_test,1:2), 'cpd_cols', LM_range(best_cpd_cols_dim_test,1:2), 'cpd_colf', LM_range(best_cpd_colf_dim_test,1:2), 'mlsvd', LM_range(best_mlsvd_dim_test,1:2));
end
