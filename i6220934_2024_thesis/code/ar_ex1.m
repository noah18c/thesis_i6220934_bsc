function [all_errors_gt1_mean, all_errors_gt2_mean] = ar_ex1(max_signals,num_experiments,optimal_order,num_components)
    addpath('./tensorlab/');
    
    % parameters for period, amplitude, and interval to be tested
    
    signal_params = [
        1, 1, 100;
        10, 1, 100;
        1, 100, 100;
        10, 100, 100;
        1, 1, 200;
        10, 1, 200;
        1, 100, 200;
        10, 100, 200
    ];
    
    %max_signals_param = size(signal_params,1);
    max_signals_param = 2;
    
    % Total number of simulations:
    % max_signals_param*max_signals*gt*num_experiments
    num_predict = 1;
    
    
    % 5 metrics, 6 models (including actual), number of different parameter
    % setups
    all_errors_gt1_mean = zeros(5,6,max_signals_param);
    all_errors_gt2_mean = zeros(5,6,max_signals_param);
    
    for sim_param = 1:max_signals_param
        
        % number of metrics by number of simulations of different signals for
        % different simulation parameters
        all_errors_gt1 = zeros(5,6,max_signals);
        all_errors_gt2 = zeros(5,6,max_signals);
    
        for sim = 1:max_signals
            disp("Parameter simulation "+sim_param+"/"+max_signals_param);
            disp("Generated signal "+sim+"/"+max_signals);
            
            % Generate sinusoidal signals
            period = signal_params(sim_param,1);
            amp = signal_params(sim_param,2);
            dt = 1;
            interval = signal_params(sim_param,3);
            t = (0:dt:interval)';
            N = length(t);  % Number of sampling points in the time series
            
            % Define base signals without decay and random addition
            base_signal1 = amp*sin(2 * pi * period * round((rand()*0.1+0.01),2) * t);
            base_signal2 = amp*sin(2 * pi * period * round((rand()*0.1+0.01),2) * t);
            base_signal3 = amp*sin(2 * pi * period * round((rand()*0.1+0.01),2) * t);
            
            % Determine option testing purposes
            signal_option = 3;
            noise_option = 1;
            
            % Apply selected option
            switch signal_option
                case 1
                    % No decay or random addition
                    signal1 = base_signal1;
                    signal2 = base_signal2;
                    signal3 = base_signal3;
                case 2
                    % No random addition
                    signal1 = base_signal1 .* exp(-0.01 * t);
                    signal2 = base_signal2 .* exp(-0.01 * t);
                    signal3 = base_signal3 .* exp(-0.01 * t);
                case 3
                    % Everything (decay and random addition)
                    signal1 = base_signal1 .* exp(-0.01 * t) + rand() * 10;
                    signal2 = base_signal2 .* exp(-0.01 * t) - rand() * 10;
                    signal3 = base_signal3 .* exp(-0.01 * t) + rand() * 10;
                case 4
                    % No decay, random addition
                    signal1 = base_signal1 + rand() * 10;
                    signal2 = base_signal2 - rand() * 10;
                    signal3 = base_signal3 + rand() * 10;
                otherwise
                    error('Invalid option selected');
            end
            
            % Combine signals to form the time series
            time_series = signal1 + signal2 + signal3;
            
            % Alternative you can check what the predictive accuracy is when
            % data is rounded, more applicable to e.g. financial data
            %time_series = round(signal1 + signal2 + signal3,3);
            
            % we simulate different ground truths (gt)
            % gt == 1: signal without noise is gt
            % gt == 2: signal with noise is gt
            
            
            for gt = 1:2    
                disp("GT "+gt);
                % Ground truth for final point
                ground_truths = zeros(num_predict, num_experiments);
                
                % Prepare storage for predictions
                predictions_full = zeros(num_predict, num_experiments);
                rel_error_full = zeros(num_predict, num_experiments);
                norm_error_full = zeros(num_predict, num_experiments);
                
                predictions_SVD = zeros(num_predict, num_experiments);
                rel_error_SVD = zeros(num_predict, num_experiments);
                norm_error_SVD = zeros(num_predict, num_experiments);
                
                predictions_CPD = zeros(num_predict, num_experiments);
                rel_error_CPD = zeros(num_predict, num_experiments);
                norm_error_CPD = zeros(num_predict, num_experiments);
                
                predictions_mlsvd = zeros(num_predict, num_experiments);
                rel_error_mlsvd = zeros(num_predict, num_experiments);
                norm_error_mlsvd = zeros(num_predict, num_experiments);
                
                predictions_cpd_col = zeros(num_predict, num_experiments);
                rel_error_cpd_col = zeros(num_predict, num_experiments);
                norm_error_cpd_col = zeros(num_predict, num_experiments);
                
                % Perform the experiment
                for experiment = 1:num_experiments 
                    if mod(experiment,10)==0
                        disp("iter "+experiment);
                    end
    
                    if gt==1
                        ground_truths(:,experiment) = time_series(end-num_predict+1:end); 
                    end
                    
                    noisy_series = time_series + noise_option*(randn(N, 1) * 0.4);
                    
                    if gt==2
                        ground_truths(:,experiment) = noisy_series(end-num_predict+1:end);
                    end
                    
                    training_series = noisy_series(1:end-num_predict);   
                    
                    
                    % 1. Full time series prediction using AR model
                    model_full = ar(training_series, optimal_order);  % AR(optimal_order) model
                    predictions_full(:,experiment) = forecast(model_full, training_series, num_predict);
                
                    % 2. Decomposition using Hankel Matrix (SVD)
                    predictions_SVD(:,experiment) = ar_svd(training_series,num_predict,optimal_order,num_components);
                
                    % 3. Decomposition using Hankel Tensor (CPD sum of mean anti-diagonals)
                    predictions_CPD(:,experiment) = ar_cpd_ms(training_series,num_predict,optimal_order,num_components);
                
                    % 4. Decomposition using Hankel Tensor (CPD predict along column c)
                    predictions_cpd_col(:,experiment) = ar_cpd_col(training_series,num_predict,10,num_components);
                
                    % 5. Decomposition using Hankel Tensor (MLSVD)
                    predictions_mlsvd(:,experiment) = ar_mlsvd(training_series,num_predict,optimal_order,0.2);
                
                    % 6. Error calculations
                    rel_error_full(:,experiment) = abs(predictions_full(:,experiment)-ground_truths(:,experiment))./abs(ground_truths(:,experiment));
                    norm_error_full(:,experiment) = norm(predictions_full(:,experiment),2)./norm(ground_truths(:,experiment),2);
                
                    rel_error_SVD(:,experiment) = abs(predictions_SVD(:,experiment)-ground_truths(:,experiment))./abs(ground_truths(:,experiment));
                    norm_error_SVD(:,experiment) = norm(predictions_SVD(:,experiment),2)./norm(ground_truths(:,experiment),2);
                
                    rel_error_CPD(:,experiment) = abs(predictions_CPD(:,experiment)-ground_truths(:,experiment))./abs(ground_truths(:,experiment));
                    norm_error_CPD(:,experiment) = norm(predictions_CPD(:,experiment),2)./norm(ground_truths(:,experiment),2);
                
                    rel_error_cpd_col(:,experiment) = abs(predictions_cpd_col(:,experiment)-ground_truths(:,experiment))./abs(ground_truths(:,experiment));
                    norm_error_cpd_col(:,experiment) = norm(predictions_cpd_col(:,experiment),2)./norm(ground_truths(:,experiment),2);
                   
                    rel_error_mlsvd(:,experiment) = abs(predictions_mlsvd(:,experiment)-ground_truths(:,experiment))./abs(ground_truths(:,experiment));
                    norm_error_mlsvd(:,experiment) = norm(predictions_mlsvd(:,experiment),2)./norm(ground_truths(:,experiment),2);
                end   
                       
                % Calculate statistics over number of experiments done
                errors_full = [mean(predictions_full,2),std(predictions_full,0,2),rmse(predictions_full,ground_truths,2),mean(rel_error_full,2),mean(norm_error_full,2)]';           
                errors_SVD = [mean(predictions_SVD,2),std(predictions_SVD,0,2),rmse(predictions_SVD,ground_truths,2),mean(rel_error_SVD,2),mean(norm_error_SVD,2)]';
                errors_CPD = [mean(predictions_CPD,2),std(predictions_CPD,0,2),rmse(predictions_CPD,ground_truths,2),mean(rel_error_CPD,2),mean(norm_error_CPD,2)]';
                errors_cpd_col = [mean(predictions_cpd_col,2),std(predictions_cpd_col,0,2),rmse(predictions_cpd_col,ground_truths,2),mean(rel_error_cpd_col,2),mean(norm_error_cpd_col,2)]';
                errors_MLSVD = [mean(predictions_mlsvd,2),std(predictions_mlsvd,0,2),rmse(predictions_mlsvd,ground_truths,2),mean(rel_error_mlsvd,2),mean(norm_error_mlsvd,2)]';
                
                % depending on ground truth, set stats to either errors_concat1 or 2
                if gt == 1
                    actual_stats = [mean(ground_truths,2),0,0,0,0]';
                    errors_concat1 = [actual_stats, errors_full, errors_SVD, errors_CPD, errors_cpd_col, errors_MLSVD];
                else
                    actual_stats = [mean(ground_truths,2),std(ground_truths,0,2),0,0,0]';
                    errors_concat2 = [actual_stats, errors_full, errors_SVD, errors_CPD, errors_cpd_col, errors_MLSVD];
                end
            end
    
            % add matrix of error calculations as a slice to 3d matrix
            all_errors_gt1(:,:,sim) = errors_concat1;
            all_errors_gt2(:,:,sim) = errors_concat2;
        end
        
        % below are the average evaluation scores (mean of matrix slices)
        % rows: mean; sd; RMSE; MRE; MRSE
        % columns: AR, SVD, CPD_ms, CPD_col, MLSVD
        % slices: results per signal parameter set
        all_errors_gt1_mean(:,:,sim_param) = mean(all_errors_gt1,3);
        all_errors_gt2_mean(:,:,sim_param) = mean(all_errors_gt2,3);
    
    end
    disp("Done!");
end