%% Data Compression Final Project
% One-bit Compressed Sensing for Image Compression
% Deepika Phadke

close all
clear all
clc

%% Load the data

% Change this location to where your MNIST data is stored. 
load('C:\Users\dphad\OneDrive\Desktop\MScSystemsAndControl\Q3\DataCompression\mnist_dataset\mnist_train.csv')
train_labels = mnist_train(:, 1);
train_data = mnist_train(:, 2:end);

%% Pre-processing

% Perform data pre-processing activities, including data preparation, ...
% sparsifying data by representing the foreground with nonzero entries ...
% and background with zeros.

% Normalise pixel values to [0,1] range
train_data_n=train_data/255.0; 

% % Sparsify (set 0 for background)
% threshold = 0.3; % assuming some noise make the distinction between foreground and background
% data_sparse = train_data_n;
% data_sparse(data_sparse <= threshold) = 0;

%% Generate the measurement matrices

% generate four fat random matrices
A25=randn(25, 784); % mean=0, std dev=1
A100=randn(100, 784); % mean=0, std dev=1
A200=randn(200, 784); % mean=0, std dev=1
A500=randn(500, 784); % mean=0, std dev=1
A1500=randn(1500, 784); % mean=0, std dev=1
Afull=randn(6272, 784); % mean=0, std dev=1

% normalise each matrix column by its unit l2 norm
A25n=A25./ vecnorm(A25, 2);
A100n=A100./ vecnorm(A100, 2);
A200n=A200./ vecnorm(A200, 2);
A500n=A500./ vecnorm(A500, 2);
A1500n=A1500./ vecnorm(A1500, 2);
Afulln=Afull./ vecnorm(Afull, 2);

%% Project vectorised images onto the A matrices
% note that each image is dimension 784. so we have 60000 images. 

y25 = sign(A25n * train_data_n'); % size 25 by 60000
y100 = sign(A100n * train_data_n'); % size 100 by 60000
y200 = sign(A200n * train_data_n'); % size 200 by 60000
y500 = sign(A500n * train_data_n'); % size 500 by 60000
y1500 = sign(A1500n * train_data_n'); % size 500 by 60000
yfull = sign(Afulln * train_data_n'); % size 500 by 60000

%% Model to recover original images

runtimes = struct();
delta = 0.01; % Step size
lambda = 0.00006; % Regularisation
tol = 1e-5; % Slightly looser convergence tolerance
max_iter = 20000; % Iterations
%[m,n]=size(data_sparse');
n=5; % only analyse 5 images otherwise it takes AN AGE to run
f_bar_prime = @(x) -x .* (x <= 0);  

%% For A25
tic
x_hat25=zeros(784,n); % preallocate x_hat recreation
A25pseudoInv=pinv(A25n); % precomputing to speed things up
hist25 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=A25pseudoInv*y25(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist25{i}, x_hat25(:,i)]=BoufanosFPI(A25n, y25(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat25=x_hat25'; % transform back to the original dimensions of x
runtimes.x_hat25 = toc;

%% For A100
tic
x_hat100=zeros(784,n); % preallocate x_hat recreation
A100pseudoInv=pinv(A100n); % precomputing to speed things up
hist100 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=A100pseudoInv*y100(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist100{i}, x_hat100(:,i)]=BoufanosFPI(A100n, y100(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat100=x_hat100'; % transform back to the original dimensions of x
runtimes.x_hat100 = toc;

%% For A200
tic
x_hat200=zeros(784,n); % preallocate x_hat recreation
A200pseudoInv=pinv(A200n); % precomputing to speed things up
hist200 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=A200pseudoInv*y200(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist200{i}, x_hat200(:,i)]=BoufanosFPI(A200n, y200(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat200=x_hat200'; % transform back to the original dimensions of x
runtimes.x_hat200 = toc;

%% For A500
tic
x_hat500=zeros(784,n); % preallocate x_hat recreation
A500pseudoInv=pinv(A500n); % precomputing to speed things up
hist500 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=A500pseudoInv*y500(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist500{i}, x_hat500(:,i)]=BoufanosFPI(A500n, y500(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat500=x_hat500'; % transform back to the original dimensions of x
runtimes.x_hat500 = toc;

%% For A1500
tic
x_hat1500=zeros(784,n); % preallocate x_hat recreation
A1500pseudoInv=pinv(A1500n); % precomputing to speed things up
hist1500 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=A1500pseudoInv*y1500(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist1500{i}, x_hat1500(:,i)]=BoufanosFPI(A1500n, y1500(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat1500=x_hat1500'; % transform back to the original dimensions of x
runtimes.x_hat1500 = toc;

%% For Afull
tic
x_hatfull=zeros(784,n); % preallocate x_hat recreation
AfullpseudoInv=pinv(Afulln); % precomputing to speed things up
histfull = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    x_hat=AfullpseudoInv*yfull(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [histfull{i}, x_hatfull(:,i)]=BoufanosFPI(Afulln, yfull(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hatfull=x_hatfull'; % transform back to the original dimensions of x
runtimes.x_hatfull = toc;

%% For A25 BIHT
tic
max_iter = 1000;  % 
tol = 5e-10;       % Convergence tolerance
lambda = 0.00005;     % Gradient weight
eta = 2;       % Learning rate

% Precompute slow things matrices
At = A25n';           % Precompute Phi'
% Generate sign measurements
y25 = sign(A25n * train_data_n'); % size 25 by 60000
x_hat25BIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHT25 = cell(n, 1);
parfor i = 1:n
    Y = diag(y25(:,i));
    % Precompute for this sample
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    YPhi = Y * A25n;
    YPhiT = YPhi';
    PhiT_y = At * y25(:,i);
    [histBIHT25{i}, x_hat25BIHT(i,:)] = optimized_BIHT(y25(:,i), A25n, YPhi, YPhiT, At, PhiT_y,  max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hat25BIHT = toc;


%% For A100 BIHT
tic

% Precompute slow things matrices
At = A100n'; % Precompute Phi'
% Generate sign measurements
y100 = sign(A100n * train_data_n'); % size 25 by 60000
x_hat100BIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHT100 = cell(n, 1);
parfor i = 1:n
    Y = diag(y100(:,i));
    % Precompute for this sample
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    YPhi = Y * A100n;
    YPhiT = YPhi';
    PhiT_y = At * y100(:,i);
    [histBIHT100{i}, x_hat100BIHT(i,:)] = optimized_BIHT(y100(:,i), A100n, YPhi, YPhiT, At, PhiT_y, max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hat100BIHT = toc;


%% For A200 BIHT
tic

% Precompute slow things matrices
At = A200n'; % Precompute Phi'
% Generate sign measurements
y200 = sign(A200n * train_data_n'); % size 25 by 60000
x_hat200BIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHT200 = cell(n, 1);
parfor i = 1:n
    Y = diag(y200(:,i));
    % Precompute for this sample
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    YPhi = Y * A200n;
    YPhiT = YPhi';
    PhiT_y = At * y200(:,i);
    [histBIHT200{i}, x_hat200BIHT(i,:)] = optimized_BIHT(y200(:,i), A200n, YPhi, YPhiT, At, PhiT_y, max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hat200BIHT = toc;

%% For A500 BIHT
tic

% Precompute slow things matrices
At = A500n'; % Precompute Phi'
% Generate sign measurements
y500 = sign(A500n * train_data_n'); % size 25 by 60000
x_hat500BIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHT500 = cell(n, 1);
parfor i = 1:n
    Y = diag(y500(:,i));
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    % Precompute for this sample
    YPhi = Y * A500n;
    YPhiT = YPhi';
    PhiT_y = At * y500(:,i);
    [histBIHT500{i}, x_hat500BIHT(i,:)] = optimized_BIHT(y500(:,i), A500n, YPhi, YPhiT, At, PhiT_y, max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hat500BIHT = toc;

%% For A1500 BIHT
tic

% Precompute slow things matrices
At = A1500n'; % Precompute Phi'
% Generate sign measurements
y1500 = sign(A1500n * train_data_n'); % size 25 by 60000
x_hat1500BIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHT1500 = cell(n, 1);
parfor i = 1:n
    Y = diag(y1500(:,i));
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    % Precompute for this sample
    YPhi = Y * A1500n;
    YPhiT = YPhi';
    PhiT_y = At * y1500(:,i);
    [histBIHT1500{i}, x_hat1500BIHT(i,:)] = optimized_BIHT(y1500(:,i), A1500n, YPhi, YPhiT, At, PhiT_y, max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hat1500BIHT = toc;

%% For Afull BIHT
tic

% Precompute slow things matrices
At = Afulln'; % Precompute Phi'
% Generate sign measurements
yfull = sign(Afulln * train_data_n'); % size 25 by 60000
x_hatfullBIHT = zeros(n, 784); % Transposed allocation to match output needs
histBIHTfull = cell(n, 1);
parfor i = 1:n
    Y = diag(yfull(:,i));
    true_sparsity = nnz(train_data_n(i,:)); % Sparsity level
    % Precompute for this sample
    YPhi = Y * Afulln;
    YPhiT = YPhi';
    PhiT_y = At * yfull(:,i);
    [histBIHTfull{i}, x_hatfullBIHT(i,:)] = optimized_BIHT(yfull(:,i), Afulln, YPhi, YPhiT, At, PhiT_y, max_iter, tol, lambda, eta, true_sparsity);
end
runtimes.x_hatfullBIHT = toc;

%% Reconstructed Images

data_info = {'A25 RFPI Image';'A100 RFPI Image';'A200 RFPI Image';'A500 RFPI Image';'A1500 RFPI Image';'A_{Full} RFPI Image';'A25 BIHT image';'A100 BIHT image';'A200 BIHT image';'A500 BIHT image';'A1500 BIHT image';'A_{Full} BIHT image'};
varnames = {'x_hat25','x_hat100','x_hat200','x_hat500','x_hat1500','x_hatfull','x_hat25BIHT','x_hat100BIHT','x_hat200BIHT','x_hat500BIHT','x_hat1500BIHT','x_hatfullBIHT'};
for i=1:length(data_info)
    % reconstructed data
    var_name = varnames{i};
    recovered_img = eval([var_name '(1, :)']); % 1x784 vector where there are 1s these will become the j loop
    img_matrix = reshape(recovered_img, [28, 28])'; % Reshape to 28x28 to visualise
    % original data
    original_img = train_data_n(1, :); % 1x784 vector
    original_img_matrix = reshape(original_img, [28, 28])'; % Reshape to 28x28 to visualise
    % plot
    figure
    subplot(1,2,1)
    imshow(original_img_matrix, []); % Display the image
    title('original image')
    subplot(1,2,2)
    imshow(img_matrix, []); % Display the image
    title(data_info{i})
end 

for i=1:length(data_info)
    % reconstructed data
    var_name = varnames{i};
    recovered_img = eval([var_name '(2, :)']); % 1x784 vector where there are 1s these will become the j loop
    img_matrix = reshape(recovered_img, [28, 28])'; % Reshape to 28x28 to visualise
    % original data
    original_img = train_data_n(2, :); % 1x784 vector
    original_img_matrix = reshape(original_img, [28, 28])'; % Reshape to 28x28 to visualise
    % plot
    figure
    subplot(1,2,1)
    imshow(original_img_matrix, []); % Display the image
    title('original image')
    subplot(1,2,2)
    imshow(img_matrix, []); % Display the image
    title(data_info{i})
end 

for i=1:length(data_info)
    % reconstructed data
    var_name = varnames{i};
    recovered_img = eval([var_name '(3, :)']); % 1x784 vector where there are 1s these will become the j loop
    img_matrix = reshape(recovered_img, [28, 28])'; % Reshape to 28x28 to visualise
    % original data
    original_img = train_data_n(3, :); % 1x784 vector
    original_img_matrix = reshape(original_img, [28, 28])'; % Reshape to 28x28 to visualise
    % plot
    figure
    subplot(1,2,1)
    imshow(original_img_matrix, []); % Display the image
    title('original image')
    subplot(1,2,2)
    imshow(img_matrix, []); % Display the image
    title(data_info{i})
end 

for i=1:length(data_info)
    % reconstructed data
    var_name = varnames{i};
    recovered_img = eval([var_name '(4, :)']); % 1x784 vector where there are 1s these will become the j loop
    img_matrix = reshape(recovered_img, [28, 28])'; % Reshape to 28x28 to visualise
    % original data
    original_img = train_data_n(4, :); % 1x784 vector
    original_img_matrix = reshape(original_img, [28, 28])'; % Reshape to 28x28 to visualise
    % plot
    figure
    subplot(1,2,1)
    imshow(original_img_matrix, []); % Display the image
    title('original image')
    subplot(1,2,2)
    imshow(img_matrix, []); % Display the image
    title(data_info{i})
end 

for i=1:length(data_info)
    % reconstructed data
    var_name = varnames{i};
    recovered_img = eval([var_name '(5, :)']); % 1x784 vector where there are 1s these will become the j loop
    img_matrix = reshape(recovered_img, [28, 28])'; % Reshape to 28x28 to visualise
    % original data
    original_img = train_data_n(5, :); % 1x784 vector
    original_img_matrix = reshape(original_img, [28, 28])'; % Reshape to 28x28 to visualise
    % plot
    figure
    subplot(1,2,1)
    imshow(original_img_matrix, []); % Display the image
    title('original image')
    subplot(1,2,2)
    imshow(img_matrix, []); % Display the image
    title(data_info{i})
end 

%% Error in estimation as a function of the number of measurements

% Define labels and colors for plotting
labels = {'Image of 5', 'Image of 0', 'Image of 4', 'Image of 1', 'Image of 9'};
colors = {'k-', 'r-', 'g-', 'b-', 'm-'};
% Call the convergence plotting function for each dataset
plot_convergence(histBIHT25, 'Convergence for BIHT A25',colors, labels);
plot_convergence(hist25, 'Convergence for RFPI A25',colors, labels);
plot_convergence(histBIHT100, 'Convergence for BIHT A100',colors, labels);
plot_convergence(hist100, 'Convergence for RFPI A100',colors, labels);
plot_convergence(histBIHT200, 'Convergence for BIHT A200',colors, labels);
plot_convergence(hist200, 'Convergence for RFPI A200',colors, labels);
plot_convergence(histBIHT500, 'Convergence for BIHT A500',colors, labels);
plot_convergence(hist500, 'Convergence for RFPI A500',colors, labels);
plot_convergence(histBIHT1500, 'Convergence for BIHT A1500',colors, labels);
plot_convergence(hist1500, 'Convergence for RFPI A1500',colors, labels);
plot_convergence(histBIHTfull, 'Convergence for BIHT A_{full}',colors, labels);
plot_convergence(histfull, 'Convergence for RFPI A_{full}',colors, labels);

%% Assessment Methods

% Display true sparsity
true_sparsity = nnz(train_data_n(5,:)); % Sparsity level
disp(['True Sparsity: ', num2str(true_sparsity)]);


% Define method/data identifiers
x_hat_list = {
    x_hat25BIHT, 'BIHT A25', A25n, y25;
    x_hat25, 'RFPI A25', A25n, y25;
    x_hat100BIHT, 'BIHT A100', A100n, y100;
    x_hat100, 'RFPI A100', A100n, y100;
    x_hat200BIHT, 'BIHT A200', A200n, y200;
    x_hat200, 'RFPI A200', A200n, y200;
    x_hat500BIHT, 'BIHT A500', A500n, y500;
    x_hat500, 'RFPI A500', A500n, y500;
    x_hat1500BIHT, 'BIHT A1500', A1500n, y1500;
    x_hat1500, 'RFPI A1500', A1500n, y1500;
    x_hatfullBIHT, 'BIHT A_{full}', Afulln, yfull;
    x_hatfull, 'RFPI A_{full}', Afulln, yfull;
};

% Number of test samples (assumes x_hat rows = number of test images)
num_samples = 5;

% Preallocate result structure
results = struct();

% Loop over each method
for k = 1:size(x_hat_list, 1)
    x_hat = x_hat_list{k, 1};
    method_name = x_hat_list{k, 2};
    A = x_hat_list{k, 3};
    y = x_hat_list{k, 4};

    % Initialize metrics
    psnr_vals = zeros(num_samples, 1);
    ssim_vals = zeros(num_samples, 1);
    mse_vals = zeros(num_samples, 1);
    support_acc = zeros(num_samples, 1);
    hamming_acc = zeros(num_samples, 1);

    for i = 1:num_samples
        x_rec = x_hat(i, :);
        x_true = train_data_n(i, :);

        psnr_vals(i) = psnr(x_rec, x_true);
        ssim_vals(i) = ssim(reshape(x_rec, 28, 28)', reshape(x_true, 28, 28)');
        mse_vals(i) = mean((x_rec - x_true).^2);

        true_support = find(x_true);
        recovered_support = find(x_rec);
        intersection = intersect(true_support, recovered_support);
        support_acc(i) = length(intersection) / length(true_support);

        y_hat = sign(A * x_rec');
        hamming_dist = sum(y_hat ~= y(:, i));
        hamming_acc(i) = 1 - hamming_dist / length(y);
    end

    % Store in results struct
    results(k).method = method_name;
    results(k).mean_psnr = mean(psnr_vals);
    results(k).mean_ssim = mean(ssim_vals);
    results(k).mean_mse = mean(mse_vals);
    results(k).mean_support_accuracy = mean(support_acc);
    results(k).mean_hamming_accuracy = mean(hamming_acc);
end

% Display results
for k = 1:length(results)
    fprintf('Method: %s\n', results(k).method);
    fprintf('  PSNR: %.2f\n', results(k).mean_psnr);
    fprintf('  SSIM: %.4f\n', results(k).mean_ssim);
    fprintf('  MSE: %.6f\n', results(k).mean_mse);
    fprintf('  Support Recovery Accuracy: %.4f\n', results(k).mean_support_accuracy);
    fprintf('  Hamming Accuracy: %.4f\n\n', results(k).mean_hamming_accuracy);
end

%% Averaged Assessment Plots
% Extract method names and metrics
method_names = {results.method};
psnr_vals = [results.mean_psnr];
ssim_vals = [results.mean_ssim];
mse_vals = [results.mean_mse];
support_vals = [results.mean_support_accuracy];
hamming_vals = [results.mean_hamming_accuracy];
fields = fieldnames(runtimes);
runtime_vals = [runtimes.(fields{1}), runtimes.(fields{7})...
    runtimes.(fields{2}), runtimes.(fields{8})...
    runtimes.(fields{3}), runtimes.(fields{9})...
    runtimes.(fields{4}), runtimes.(fields{10})...
    runtimes.(fields{5}), runtimes.(fields{11})...
    runtimes.(fields{6}), runtimes.(fields{12})];

% Separate BIHT and RFPI (assuming alternating order)
biht_idx = 1:2:length(method_names);
rfpi_idx = 2:2:length(method_names);

x_vals = 1:length(biht_idx);  % Just one x-axis per method type (A25, A100, etc.)

% Function to plot both lines per metric
plot_metric = @(vals, title_str, ylbl) ...
    plot_dual_lines(x_vals, vals(biht_idx), vals(rfpi_idx), title_str, ylbl, {'A25','A100','A200','A500','A1500','A_{full}'});

% Plot in a tiled layout
figure;
tiledlayout(3,2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
plot_metric(psnr_vals, 'PSNR', 'Value');

nexttile;
plot_metric(ssim_vals, 'SSIM', 'Value');

nexttile;
plot_metric(mse_vals, 'MSE', 'Value');

nexttile;
plot_metric(support_vals, 'Support Accuracy', 'Value');

nexttile;
plot_metric(hamming_vals, 'Hamming Accuracy', 'Value');

nexttile;
plot_metric(runtime_vals, 'Run Times', 'Time [s]')

% -------- Helper function --------
function plot_dual_lines(x, biht_vals, rfpi_vals, title_str, ylbl, xlabels)
    plot(x, biht_vals, '-or', 'LineWidth', 2, 'DisplayName', 'BIHT'); hold on;
    plot(x, rfpi_vals, '-ob', 'LineWidth', 2, 'DisplayName', 'RFPI');
    title(title_str);
    ylabel(ylbl);
    xticks(x);
    xticklabels(xlabels);
    xtickangle(45);
    grid on;
    %legend('Location', 'best');
end

%% Full Assessment Plots
% % Extract method names and metrics
% method_names = {results.method};
% 
% % Preallocate arrays for all samples
% all_psnr = zeros(length(results), num_samples);
% all_ssim = zeros(length(results), num_samples);
% all_mse = zeros(length(results), num_samples);
% all_support = zeros(length(results), num_samples);
% all_hamming = zeros(length(results), num_samples);
% 
% for k = 1:length(results)
%     x_hat = x_hat_list{k, 1};
%     A = x_hat_list{k, 3};
%     y = x_hat_list{k, 4};
% 
%     % Calculate metrics for all samples
%     for i = 1:num_samples
%         x_rec = x_hat(i, :);
%         x_true = train_data_n(i, :);
% 
%         all_psnr(k,i) = psnr(x_rec, x_true);
%         all_ssim(k,i) = ssim(reshape(x_rec, 28, 28)', reshape(x_true, 28, 28)');
%         all_mse(k,i) = mean((x_rec - x_true).^2);
% 
%         true_support = find(x_true);
%         recovered_support = find(x_rec);
%         intersection = intersect(true_support, recovered_support);
%         all_support(k,i) = length(intersection) / length(true_support);
% 
%         y_hat = sign(A * x_rec');
%         hamming_dist = sum(y_hat ~= y(:, i));
%         all_hamming(k,i) = 1 - hamming_dist / length(y);
%     end
% end
% 
% % Separate BIHT and RFPI
% biht_idx = 1:2:length(method_names);
% rfpi_idx = 2:2:length(method_names);
% 
% x_vals = 1:length(biht_idx);  % x-axis positions for methods
% 
% % Create color map for samples
% sample_colors = lines(num_samples); % Different color for each sample
% sample_labels = arrayfun(@(x) sprintf('Sample %d', x), 1:num_samples, 'UniformOutput', false);
% 
% % Function to plot all samples as separate lines
% plot_individual_samples = @(vals, title_str, ylbl) ...
%     plot_sample_lines(x_vals, vals, biht_idx, rfpi_idx, title_str, ylbl, ...
%                      {'A25','A100','A200','A500','A1500','A_{full}'}, ...
%                      sample_colors, sample_labels);
% 
% % Plot in a tiled layout
% figure;
% tiledlayout(3,2, 'Padding', 'compact', 'TileSpacing', 'compact');
% 
% nexttile;
% plot_individual_samples(all_psnr, 'PSNR', 'Value');
% 
% nexttile;
% plot_individual_samples(all_ssim, 'SSIM', 'Value');
% 
% nexttile;
% plot_individual_samples(all_mse, 'MSE', 'Value');
% 
% nexttile;
% plot_individual_samples(all_support, 'Support Accuracy', 'Value');
% 
% nexttile;
% plot_individual_samples(all_hamming, 'Hamming Accuracy', 'Value');
% 
% % Runtime plot remains the same (only mean values)
% nexttile;
% plot_metric(runtime_vals, 'Run Times', 'Time [s]');
% 
% % -------- New helper function --------
% function plot_sample_lines(x, vals, biht_idx, rfpi_idx, title_str, ylbl, xlabels, colors, labels)
%     hold on;
% 
%     % Plot BIHT lines for each sample
%     for sample = 1:size(vals,2)
%         biht_vals = vals(biht_idx, sample)';
%         plot(x, biht_vals, '-o', 'Color', colors(sample,:), ...
%             'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', ['BIHT ' labels{sample}]);
%     end
% 
%     % Plot RFPI lines for each sample
%     for sample = 1:size(vals,2)
%         rfpi_vals = vals(rfpi_idx, sample)';
%         plot(x, rfpi_vals, '--s', 'Color', colors(sample,:), ...
%             'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', ['RFPI ' labels{sample}]);
%     end
% 
%     title(title_str);
%     ylabel(ylbl);
%     xticks(x);
%     xticklabels(xlabels);
%     xtickangle(45);
%     grid on;
%     %legend('Location', 'bestoutside');
% end

%% Functions

% function for Boufanos and Baraniuk Algorithm #1 

function [record, x_hat] = BoufanosFPI(A, y, x_hat, delta, lambda, tol, max_iter,f_bar_prime)

    % [1] P. T. Boufounos and R. G. Baraniuk, "1-bit compressive sensing," 
    %     in Proceedings of the 42nd Annual Conference on Information Sciences and Systems (CISS), 
    %     Princeton, NJ, USA, 2008, pp. 16–21. doi: 10.1109/CISS.2008.4558487.
    %     Algorithm 1: Renormalized Fixed Point Iteration.

    % Inputs
    % A                 - measurement matrix
    % y                 - measurements
    % xhat              - estimation
    % delta             - step size
    % lambda            - relaxation parameter
    % tolerance         - convergence tolerance stopping criterion
    % max_iter          - backup stopping criteria to prevent endless looping
    % f_bar_prime       - anonymous function for the algorithm below

    % Output
    % x_hat             - reconstructed signal

    % Initialize history log
    record.obj = zeros(max_iter, 1);
    record.diff = zeros(max_iter, 1);
    
    x_hat_new = 10*x_hat; % force the loop to start 
    % function for f_bar_prime
    %f_bar_prime = @(x) -x .* (x <= 0); % anonymous function for the algorithm below
    
    % 2 Loop Start and Tolerance Setting
    k=0;
    
    while norm(x_hat_new - x_hat) >= tol
    
        if k>max_iter
            break 
        end
        
        % update
        x_hat = x_hat_new;
        k=k+1;
    
        
        % 3 One-sided Quadratic Gradient
        residual = y .* (A * x_hat);  % Equivalent to Y*A*x_hat (no diag saves time)
        fbar_k=A' * (y .* f_bar_prime(residual));
    
        % 4 Gradient Projection on Sphere Surface
        f_tilda_k=fbar_k-dot(fbar_k, x_hat)*x_hat;
    
        % 5 One-sided Quadratic Gradient Descent
        h=x_hat-delta*f_tilda_k;
    
        % 6 Shrinkage (l1 gradient descent)
        u=sign(h) .* max(abs(h) - delta * lambda, 0);
    
        % Normalisation
        u_norm = norm(u);
        x_hat_new = u / u_norm;
        fprintf('Iter %d: residual norm = %f, change = %f\n', k, norm(residual), norm(x_hat_new - x_hat));

        record.obj(k) = sum(sign(A * x_hat_new) ~= sign(A * x_hat)); % sign mismatches
        record.diff(k) = norm(x_hat_new - x_hat);
    
    end
    
    record.iter = k;
    % Display results
    fprintf('Converged in %d iterations.\n', k);
    % disp('Final estimate:');
    % disp(x_hat_new);

end

% BIHT

function [record, x_hat] = optimized_BIHT(y, Phi, YPhi, YPhiT, PhiT, PhiT_y, max_iter, tol, lambda, eta, K)

    % [2] Z. Li, W. Xu, X. Zhang, and J. Lin, "A survey on one-bit compressed sensing: theory and applications," 
    %     *Frontiers of Computer Science*, vol. 12, no. 2, pp. 217–230, 2018. 
    %     doi: 10.1007/s11704-017-6132-7.

    % Inputs
    % y             - measurements
    % Phi           - measurement matrix
    % YPhi          - precomputed multiplication of Phi with diag(y)
    % YPhiT         - precomputed result of (Y*Phi)^T for speed
    % PhiT          - precomputed result of (Phi)^T for speed
    % PhiT_y        - precomputed result of (Phi^T)*y for speed
    % lambda        - gradient weighting factor
    % tol           - convergence tolerance stopping criterion
    % max iter      - backup stopping criteria to prevent endless looping
    % eta           - learning rate
    % K             - Sparsity level (number of nonzero entries to retain)
    
    % Output    
    % x_hat         - reconstructed signal
    % record        - record of convergence per iteration


    % Initialize with better starting point
    x_prev = zeros(size(PhiT_y));
    idx = randperm(length(x_prev), K);       % Pick K random indices
    x_prev(idx) = randn(K, 1);               % Random values on those indices
    x_prev = x_prev / norm(x_prev);          % Normalize to unit sphere

    % Initialize history log
    record.obj = zeros(max_iter, 1);
    record.diff = zeros(max_iter, 1);
    
    for iter = 1:max_iter
        % Compute gradient components
        margin = YPhi * x_prev;
        neg_margin = min(0, margin);                  % Efficient one-sided loss
        grad_l2 = (1/2) * (YPhiT) * neg_margin;
        grad_l1 = (1/2) * (PhiT_y - PhiT * sign(Phi * x_prev));
        % grad = lambda * grad_l2 + lambda * grad_l1;
        sign_loss = lambda * sum(sign(Phi * x_prev) ~= y);  % Penalize sign mismatches
        grad = grad_l2 + grad_l1 + sign_loss;  % Total gradient with added sign loss
        
        % Update with hard thresholding
        u = x_prev + eta * grad;
        
        % Sparsity projection: retain top-K entries - addition by me not in
        % original algorithm
        [~, idx] = maxk(abs(u), K);
        x_new = zeros(size(u));
        x_new(idx) = u(idx);

        % Normalize
        x_new = x_new / max(norm(x_new), 1e-8);

        % Track convergence
        diff = norm(x_new - x_prev);
        % sign mismatches
        record.obj(iter) = sum(sign(Phi * x_new) ~= sign(Phi * x_prev)); 
        % difference (error)
        record.diff(iter) = diff;

        eta = eta / (1 + iter / 1000);  %  - addition by me not in
        % original algorithm: gradually reduce eta to fine-tune the result

        if diff < tol 
            break;  
        end

        x_prev = x_new;
    end

    % Output
    x_hat = x_new;
    record.iter = iter;
    fprintf('BIHT converged in %d iterations.\n', iter);
end


% Helper function to plot convergence data
function plot_convergence(hist_list, title_prefix, colors, labels)
    figure;
    hold on;
    for i = 1:length(hist_list)
        plot(1:hist_list{i}.iter, hist_list{i}.diff(1:hist_list{i}.iter), colors{i}, 'DisplayName', labels{i});
    end
    ylabel('Change');
    xlabel('Iteration');
    title([title_prefix ' - Change in xhat']);
    legend;
    hold off;

    figure;
    hold on;
    for i = 1:length(hist_list)
        plot(1:hist_list{i}.iter, hist_list{i}.obj(1:hist_list{i}.iter), colors{i}, 'DisplayName', labels{i});
    end
    ylabel('Sign Mismatches');
    xlabel('Iteration');
    title([title_prefix ' - Sign Mismatch']);
    legend;
    hold off;
end