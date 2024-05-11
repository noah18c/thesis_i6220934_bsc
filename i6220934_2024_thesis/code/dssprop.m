function dssprop(state_space)
    A = state_space.A;
    B = state_space.B;
    C = state_space.C;
    D = state_space.D;

    disp("State-space properties: ")
    disp(["Number of states: "+size(A,1); ...
        "Number of inputs: "+size(B,2); ...
        "Number of outputs: "+size(C,1)])
    
    disp("State-space dimensions: ")
    disp(["A_rows:"+size(A,1),"A_cols:"+size(A,2); ...
        "B_rows:"+size(B,1),"B_cols:"+size(B,2); ...
        "C_rows:"+size(C,1),"C_cols:"+size(C,2); ...
        "D_rows:"+size(D,1),"D_cols:"+size(D,2)])

    disp("Max absolute eigen value: "+max(abs(eig(A))))

    disp("Eigen values of A: ")
    disp(abs(eig(A)))
end