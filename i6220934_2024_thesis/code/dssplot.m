function plt = dssplot(state_space, maxTime, input, init)
    statenum = size(state_space.A,1);
    inputnum = size(state_space.B,2);

    if ~exist('init', 'var')
        % if there are no initial conditions, then set it to random
        init = rand(statenum,1);
    end

    if ~exist('input','var')
        % if there is no input, then set it to zero
        input = zeros(inputnum,1);
    end
    
    [~,y] = dsssim(state_space,maxTime,input, init);
    
    plt = plot(0:maxTime,y);
    xlabel("Time until t="+ (maxTime)+" ");
    ylabel("Output");
    title("Impulse response of state-space system with "+ statenum+ " states");
    legend('show');
end