function [] = state_space_plotter(state_space, input, maxTime)
    A = state_space.A;
    B = state_space.B;
    C = state_space.C;
    D = state_space.D;

    statenum = size(A,1);
    outputnum = size(C,1);

    x = zeros(statenum,maxTime);
    y = zeros(outputnum,maxTime);
    x(:,1) = rand(statenum,1);
    y(:,1) = C*x(:,1);
    
    for i=2:maxTime
        y(:,i) = C*x(:,i-1) + D*;
        x(:,i) = A*x(:,i-1);
    end
    
    plot(0:maxTime-1,y)
    xlabel("Time until t="+ (maxTime-1)+" ");
    ylabel("Output");
    title("Impulse response of state-space system with "+ statenum+ " states");
    legend('show');
end