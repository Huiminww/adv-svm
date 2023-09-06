clear all; clc;
global iters;
global batch_size;
batch_size = 500;
reg_param = 1e-6;
step_size = 1e-6;
C_epsilon = 1;

load('./data/cifar_1v9.mat');

x=find(trainlabel==1);
trainlabel(x)=0;
x=find(trainlabel==9);
trainlabel(x)=1;

x=find(testlabel==1);
testlabel(x)=0;
x=find(testlabel==9);
testlabel(x)=1;

sequence = 'unif';
strategy = 'diminishing';
k=2;

ntr = size(traindata,1);
nte = size(testdata,1);

rp = randperm(ntr);
traindata = traindata(rp, :);
trainlabel = trainlabel(rp,:);
rp = randperm(nte);
testdata = testdata(rp,:);
testlabel = testlabel(rp,:);

traindata=traindata';
testdata=testdata';

traindata = im2double(traindata);
testdata = im2double(testdata);

normtrain = sqrt(sum(traindata.^2, 1));
normtest = sqrt(sum(testdata.^2, 1));

traindata = bsxfun(@rdivide, traindata, normtrain);
testdata = bsxfun(@rdivide, testdata, normtest);

iters = floor(5*(ntr/batch_size));

% PCA on data. 
fprintf('-- pca of data ...\n');
subsample_size = 1e4;
subsample_idx = randsample(ntr, subsample_size);
covmat = traindata(:, subsample_idx) * traindata(:, subsample_idx)' ./ subsample_size; 
opts.isreal = true; 
pca_dim = 100; 
[v, ss] = eigs(double(covmat), pca_dim, 'LM', opts); 

traindata = v' * traindata; 
testdata = v' * testdata; 
            
trainY = zeros(2, ntr, 'single');
tl_idx = sub2ind([2, ntr], trainlabel+1,(1:ntr)');
trainY(tl_idx) = 1;
testY = zeros(2, nte, 'single');
tl_idx = sub2ind([2, nte], testlabel+1, (1:nte)');
testY(tl_idx) = 1;

[train_error_mat, test_error_mat, time]...
         =adversarial_training(traindata, trainlabel, testdata, testlabel, trainY, testY, ntr, nte, k, reg_param, step_size, sequence,strategy, C_epsilon);
            
output_path='./ad_cifar1v9_normal.mat';
save(output_path, 'test_error_mat', 'time');


    

