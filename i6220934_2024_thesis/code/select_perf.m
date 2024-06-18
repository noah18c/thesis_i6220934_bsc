function [minimals, indices] = select_perf(experiment, measure)
    perf = mean(experiment,3);
    perfMeasure = squeeze(perf(measure,:,:,:))';
    [minimals, indices] = min(perfMeasure(:,2:end),[],1);
end