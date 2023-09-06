% computing quasi-Monte Carlo feature maps for Gaussian kernel
% Inputs:
%   X: input matrix with size n by d
%   sigma: bandwidth in Gaussian kernel
%   s: size of the feature map
%   sequence: a string that denotes which low-discrepancy sequence to use
%   W: build features maps using points from previous constructions (ignore it to compute new feature maps)
%   scram: scramble the sequence or not
%   skip, leap: properties in sequences that might affect the final accuracy
% Outputs:
%   Z: new feature matrix with size n by s and the elements are complex
%   W: points used in the construction that can be used in future construction

function phix = random_fourier_qmc_ns(X, sigma, s, sequence, W, scram, skip, leap)
global p;
global points;
global points_2;
m=400;

%d: dimension; n: lines
[d,~] = size(X);
 

if isempty(W)
    if strcmp(sequence, 'unif')  
        randn('seed',skip);
        W = randn(d,s);
        tmp = W'*X;
        phix=[cos(tmp); sin(tmp)] ./sqrt(s);
    else
        switch sequence
          case 'halton'
            p = haltonset(d,'Skip',skip,'Leap',leap);
            if scram p = scramble(p,'RR2'); end
            points = p(1:s,1:d);
            points_2 = p((s+m+1):(2*s+m),1:d);
        case 'sobol'
            p = sobolset(d,'Skip',skip,'Leap',leap);
            if scram p = scramble(p,'MatousekAffineOwen'); end
            points = p(1:s,1:d);
            points_2 = p((s+m+1):(2*s+m),1:d);
        case 'lattice'
            latticeseq_b2('initskip'); % see http://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/
            points = latticeseq_b2(d,s,2^20,skip,leap)';
            points_2 = latticeseq_b2(d,s,2^20,skip+1,leap)';
        case 'digit'
            load sobol_Cs.col % see http://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/
            %load nxs32m32.col
            digitalseq_b2g('initskip', sobol_Cs, skip); 
            points = digitalseq_b2g(d,s)';
            points_2 = digitalseq_b2g(d,s)';
        end
        W = norminv(points', 0, 1);
        W_2=norminv(points_2',0,1);
        tmp = W'*X;
        tmp2 = W_2'*X;
        phix=[cos(tmp)+cos(tmp2); sin(tmp)+sin(tmp2)] ./sqrt(4*s);
    end
end

end