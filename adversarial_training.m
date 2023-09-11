function[train_error_mat, test_error_mat, time]...
    =adversarial_training(traindata, trainlabel, testdata, testlabel, trainY, testY, ntr, nte, k, reg_param, step_size, sequence,strategy,C_epsilon)

global f;
global iters;
global batch_size;

train_true_y = trainlabel';
test_true_y = testlabel';

% Find the median pairwise distance.
randn('seed', 1);
rperm = randperm(ntr);
dismat = 1;
s_coeff = 1;
s = 1 ./ (s_coeff * median(dismat)).^2

step_size0 = 1;
step_size1 = step_size;

n = 2^24
blocksz = 1500

scram=0;
%generate primes
prime=primes(4000);

f = zeros(1,iters);

train_error_mat(iters) = 0;
test_error_mat(iters) = 0;
t = 0;
time=zeros(1,iters);

W = zeros(k, 2*n);

batch_idx = [1:batch_size];
test_preds = zeros(k, nte);
train_pred_1 = zeros(1,batch_size);

global C
global KTYPE
global KSCALE
C = 10;
KTYPE = 6;
KSCALE = 3;

[d,~] = size(traindata);
loss = zeros(batch_size,d);

for j = 1:iters
    fprintf('--iters no %d\n', j);
    
    t1=clock;
    batch_idx = mod(batch_idx + batch_size - 1, ntr) + 1;
    batch_data = traindata(:, batch_idx);
    batch_label = trainlabel';
    batch_label = batch_label(:, batch_idx);
    batch_data = batch_data';
    batch_label = batch_label';
    
    batch_data = batch_data';
    
    f_idx = j - 1;
    t2=clock;
    
    testX=random_fourier_qmc_ns(testdata,s,blocksz,sequence,[],scram,(f_idx+1)*43,(f_idx+1)*prime(j));
    
    t3=clock;
    w_idx = f_idx*2*blocksz+1:(f_idx+1)*2*blocksz;
    train_batch_X=random_fourier_qmc_ns(batch_data,s,blocksz,sequence,[],scram,(f_idx+1)*43,(f_idx+1)*prime(j));
    
    train_batch_preds = zeros(k, batch_size);
    for inner_j = 0:f_idx-1
        inner_w_idx = inner_j*2*blocksz+1:(inner_j+1)*2*blocksz;
        train_batch_preds = train_batch_preds + ...
            W(:,inner_w_idx) * random_fourier_qmc_ns(batch_data,s,blocksz,sequence,[],scram,(inner_j+1)*43,(inner_j+1)*prime(inner_j+1));
    end
    residue = -C*trainY(:,batch_idx);
    
    %converge
    covx = train_batch_X * train_batch_X' / batch_size;
    preconditioner = covx + (reg_param + 1e-7) * eye(2*blocksz); %%how to process this variable
    if strcmp(strategy,'diminishing')
        step_size = step_size0 / (1 + step_size1 * j);
    elseif strcmp(strategy,'constant')
        step_size = step_size1;
    end

    %alpha_i
    updateW =  -step_size * (residue * train_batch_X'/ batch_size + reg_param * W(:, w_idx)) / preconditioner;
    %save the value of alpha_i
    W(:, w_idx) = W(:, w_idx) + updateW;   
    
    %compute alpha_j
    for inner_j = 0:f_idx-1  
        inner_w_idx = inner_j*2*blocksz+1:(inner_j+1)*2*blocksz;
        W(:, inner_w_idx) = (1 - step_size * (1+C_epsilon/f(inner_j+1))) * W(:, inner_w_idx);
    end
    train_preds_batch = train_batch_preds + updateW * train_batch_X;
    
    t4=clock;
    t = t + etime(t2,t1)+etime(t4,t3);
    time(j)=t;
    
    [~, train_pred_y] = max(train_preds_batch, [], 1);
    train_pred_y = train_pred_y-1;
    
    for i = 1 : batch_size
        if train_pred_y(i) == 0
            train_pred_1(i) = -train_preds_batch(1,i);
        else
            train_pred_1(i) = train_preds_batch(2,i);
        end
    end
    
    inv_matrix = inv(train_batch_X'*train_batch_X)*train_batch_X';
    f(j) = norm(train_pred_1*inv_matrix,2);
    
    train_error = sum(train_pred_y ~= train_true_y(batch_idx)) / batch_size;
    
    test_preds = test_preds + updateW * testX;
    [~, test_pred_y] = max(test_preds, [], 1);
    test_pred_y = test_pred_y-1;
    test_error = sum(test_pred_y ~= test_true_y) / nte;
    
    fprintf('---running time: %f\n',t)
    
    fprintf('---step size: %f\n', step_size)
    
    fprintf('---reg_param: %f\n', reg_param)
    
    test_error_mat(j) = test_error;

    fprintf('---test error: %f\n', test_error)
    
end
