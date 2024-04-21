function [x,y] = dsssim(state_space, maxTime, input, init)
    A = state_space.A;
    B = state_space.B;
    C = state_space.C;
    D = state_space.D;

    statenum = size(A,1);
    inputnum = size(B,2);
    outputnum = size(C,1);

    x = zeros(statenum,maxTime+1);
    y = zeros(outputnum,maxTime+1);

    if ~exist('init', 'var')
        % if there are no initial conditions, then set it to random
        x(:,1) = rand(statenum,1);
    else
        x(:,1) = init;
    end

    if ~exist('input','var')
        % if there is no input, then set it to zero
        input = zeros(inputnum,1);
    end

    y(:,1) = C*x(:,1);
    
    %state updates
    for i=2:maxTime+1
        y(:,i) = C*x(:,i-1) + D*input;
        x(:,i) = A*x(:,i-1) + B*input;
    end
end