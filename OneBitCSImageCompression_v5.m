%% Data Compression Final Project
% One-bit Compressed Sensing for Image Compression
% Deepika Phadke

close all
clear all
clc

%% Load the data

load('C:\Users\dphad\OneDrive\Desktop\MScSystemsAndControl\Q3\DataCompression\mnist_dataset\mnist_train.csv')
train_labels = mnist_train(:, 1);
train_data = mnist_train(:, 2:end);

%% Pre-processing

% Perform data pre-processing activities, including data preparation, ...
% sparsifying data by representing the foreground with nonzero entries ...
% and background with zeros.

% Normalise pixel values to [0,1] range
train_data_n=train_data/255.0; 

% Sparsify (set 0 for background)
threshold = 0.3; % assuming some noise make the distinction between foreground and background
data_sparse = train_data_n;
data_sparse(data_sparse <= threshold) = 0;

% missed detection vs false alarms look up ROC - yet to check/do
% recommended by Tutor

%% Small test to check things are working

sample_img = train_data_n(1, :); % 1x784 vector
img_matrix = reshape(sample_img, [28, 28])'; % Reshape to 28x28 to visualise
figure
imshow(img_matrix, []); % Display the image
title('original image')
threshold = graythresh(img_matrix); % Calculate Otsu's threshold
x_sparse = sample_img;
x_sparse(x_sparse <= threshold) = 0;
fprintf('Otsu''s automatic threshold: %.3f\n', threshold);
img_matrix_sparse = reshape(x_sparse, [28, 28])'; % Reshape to 28x28 to visualise
figure
imshow(img_matrix_sparse, []); % Display the image
title('sparsified image')


%% Generate the measurement matrices

% generate four fat random matrices
A25=randn(25, 784); % mean=0, std dev=1
A100=randn(100, 784); % mean=0, std dev=1
A200=randn(200, 784); % mean=0, std dev=1
A500=randn(500, 784); % mean=0, std dev=1

% normalise each matrix column by its unit l2 norm
A25n=A25./ vecnorm(A25, 2);
A100n=A100./ vecnorm(A100, 2);
A200n=A200./ vecnorm(A200, 2);
A500n=A500./ vecnorm(A500, 2);

%% Project vectorised images onto the A matrices
% note that each image is dimension 784. so we have 60000 images. 

y25 = sign(A25n * data_sparse'); % size 25 by 60000
y100 = sign(A100n * data_sparse'); % size 100 by 60000
y200 = sign(A200n * data_sparse'); % size 200 by 60000
y500 = sign(A500n * data_sparse'); % size 500 by 60000

%% Model to recover original images

delta = 0.01;      % Step size
lambda = 0.00006;    % Regularisation
tol = 1e-5;        % Slightly looser convergence tolerance
max_iter = 20000; % Iterations
%[m,n]=size(data_sparse');
n=5;
f_bar_prime = @(x) -x .* (x <= 0);  

%% For A25
tic
x_hat25=zeros(784,n); % preallocate x_hat recreation
A25pseudoInv=pinv(A25n); % precomputing to speed things up
hist25 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    %Y = diag(y25(:,i)); % Note that Y is not y. Y is the product of each quantized measurement with the measurement and is always non-negative
    x_hat=A25pseudoInv*y25(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist25{i}, x_hat25(:,i)]=BoufanosFPI(A25n, y25(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat25=x_hat25'; % transform back to the original dimensions of x

%% For A100
x_hat100=zeros(784,n); % preallocate x_hat recreation
A100pseudoInv=pinv(A100n); % precomputing to speed things up
hist100 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    %Y = diag(y100(:,i)); % Note that Y is not y. Y is the product of each quantized measurement with the measurement and is always non-negative
    x_hat=A100pseudoInv*y100(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist100{i}, x_hat100(:,i)]=BoufanosFPI(A100n, y100(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat100=x_hat100'; % transform back to the original dimensions of x

%% For A200
x_hat200=zeros(784,n); % preallocate x_hat recreation
A200pseudoInv=pinv(A200n); % precomputing to speed things up
hist200 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    %Y = diag(y200(:,i)); % Note that Y is not y. Y is the product of each quantized measurement with the measurement and is always non-negative
    x_hat=A200pseudoInv*y200(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist200{i}, x_hat200(:,i)]=BoufanosFPI(A200n, y200(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat200=x_hat200'; % transform back to the original dimensions of x

%% For A500
x_hat500=zeros(784,n); % preallocate x_hat recreation
A500pseudoInv=pinv(A500n); % precomputing to speed things up
hist500 = cell(n, 1);
parfor i=1:n
    % Step 1 Initialisation:
    %Y = diag(y500(:,i)); % Note that Y is not y. Y is the product of each quantized measurement with the measurement and is always non-negative
    x_hat=A500pseudoInv*y500(:,i); % per paper this is a good recommended starting point
    x_hat = x_hat / norm(x_hat);  % Normalise to unit norm
    % Steps 2-8 of Algorithm:
    [hist500{i}, x_hat500(:,i)]=BoufanosFPI(A500n, y500(:,i), x_hat, delta, lambda, tol, max_iter,f_bar_prime);
end
x_hat500=x_hat500'; % transform back to the original dimensions of x
toc

%% For A25 BIHT
tic

max_iter = 1000;  % 
tol = 5e-10;       % Convergence tolerance
lambda = 0.00005;     % Gradient weight
eta = 2;       % Learning rate

% Precompute slow things matrices
At = A25n';           % Precompute Phi'
% Generate sign measurements
y25 = sign(A25n * data_sparse'); % size 25 by 60000
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
toc

%% For A100 BIHT
tic

% Precompute slow things matrices
At = A100n';           % Precompute Phi'
% Generate sign measurements
y100 = sign(A100n * data_sparse'); % size 25 by 60000
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
toc


%% For A200 BIHT
tic

% Precompute slow things matrices
At = A200n';           % Precompute Phi'
% Generate sign measurements
y200 = sign(A200n * data_sparse'); % size 25 by 60000
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
toc

%% For A500 BIHT
tic

% Precompute slow things matrices
At = A500n';           % Precompute Phi'
% Generate sign measurements
y500 = sign(A500n * data_sparse'); % size 25 by 60000
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
toc

%% Reconstructed Images

data_info = {'A25 min(||x||_1) Image';'A100 min(||x||_1) Image';'A200 min(||x||_1) Image';'A500 min(||x||_1) Image';'A25 BIHT image';'A200 BIHT image';'A100 BIHT image';'A500 BIHT image'};
varnames = {'x_hat25','x_hat100','x_hat200','x_hat500','x_hat25BIHT','x_hat100BIHT','x_hat200BIHT','x_hat500BIHT'};
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

true_sparsity = nnz(data_sparse(5,:)); % Sparsity level

% Extract the individual convergence histories
hist1 = histBIHT500{1};
hist2 = histBIHT500{2};
hist3 = histBIHT500{3};
hist4 = histBIHT500{4};
hist5 = histBIHT500{5};
hist6 = hist500{1};
hist7 = hist500{2};
hist8 = hist500{3};
hist9 = hist500{4};
hist10 = hist500{5};


% Create a figure and hold on to plot all histograms on the same plot
figure;
hold on;

plot(1:hist1.iter, hist1.diff(1:hist1.iter), 'b', 'DisplayName', 'Image of 5');
plot(1:hist2.iter, hist2.diff(1:hist2.iter), 'r', 'DisplayName', 'Image of 0');
plot(1:hist3.iter, hist3.diff(1:hist3.iter), 'g', 'DisplayName', 'Image of 4');
plot(1:hist4.iter, hist4.diff(1:hist4.iter), 'm', 'DisplayName', 'Image of 1');
plot(1:hist5.iter, hist5.diff(1:hist5.iter), 'c', 'DisplayName', 'Image of 9');
xlabel('Iteration');
ylabel('Change');
title('Convergence for BIHT A500');
legend;


% Create a figure and hold on to plot all histograms on the same plot
figure;
hold on;

plot(1:hist6.iter, hist6.diff(1:hist6.iter), 'b', 'DisplayName', 'Image of 5');
plot(1:hist7.iter, hist7.diff(1:hist7.iter), 'r', 'DisplayName', 'Image of 0');
plot(1:hist8.iter, hist8.diff(1:hist8.iter), 'g', 'DisplayName', 'Image of 4');
plot(1:hist9.iter, hist9.diff(1:hist9.iter), 'm', 'DisplayName', 'Image of 1');
plot(1:hist10.iter, hist10.diff(1:hist10.iter), 'c', 'DisplayName', 'Image of 9');
xlabel('Iteration');
ylabel('Change');
title('Convergence for L1 Norm Min A500');
legend;


% Display true sparsity
true_sparsity = nnz(data_sparse(5,:)); % Sparsity level
disp(['True Sparsity: ', num2str(true_sparsity)]);




% Yet to be done! Start from here next time.

% % (1) Column norms (should all be ≈1)
% column_norms = vecnorm(A25n);
% fprintf('Norm mean: %f, std: %f\n', mean(column_norms), std(column_norms));
% 
% % (2) Coherence (should be < 0.5 for m=25)
% gram = abs(A25n' * A25n);
% gram(logical(eye(size(gram)))) = 0;
% fprintf('Coherence: %f\n', max(gram(:)));
% 
% % (3) Condition number (should be < 1e3)
% fprintf('Condition number: %f\n', cond(A25n));


psnr_val = psnr(x_hat500BIHT(1,:), data_sparse(1,:));
true_support = find(data_sparse(1,:));
recovered_support = find(x_hat500BIHT);
intersection = intersect(true_support, recovered_support);
support_recovery_accuracy = length(intersection) / length(true_support);


ssim_val = ssim(reshape(x_hat500BIHT(1,:), 28, 28)', reshape(data_sparse(1,:), 28, 28)');

mse = mean((x_hat500BIHT(1,:) - data_sparse(1,:)).^2);

y_hat = sign(A500n * x_hat500BIHT(1,:)');
hamming_dist = sum(y_hat ~= y500(:,1));
hamming_accuracy = 1 - hamming_dist / length(y500(:,1));

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
    % x_prev = PhiT_y + 0.8 * randn(size(PhiT_y));  % Slight perturbation
    % x_prev = x_prev / norm(x_prev);

    x_prev = zeros(size(PhiT_y));
    idx = randperm(length(x_prev), K);       % Pick K random indices
    x_prev(idx) = randn(K, 1);               % Random values on those indices
    x_prev = x_prev / norm(x_prev);          % Normalize to unit sphere

    % x_temp = PhiT_y;
    % [~, idx] = maxk(abs(x_temp), K);
    % x_prev = zeros(size(x_temp));
    % x_prev(idx) = x_temp(idx);
    % x_prev = x_prev / norm(x_prev);

    % x_prev = randn(size(PhiT_y));    % Start with more random values
    % x_prev = x_prev / norm(x_prev);  % Normalize to unit vector

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
        
        % Sparsity projection: retain top-K entries
        [~, idx] = maxk(abs(u), K);
        x_new = zeros(size(u));
        x_new(idx) = u(idx);

        % Normalize
        x_new = x_new / max(norm(x_new), 1e-8);

        % Track convergence
        diff = norm(x_new - x_prev);
        record.obj(iter) = sum(sign(Phi * x_new) ~= sign(Phi * x_prev)); % sign mismatches
        record.diff(iter) = diff;

        eta = eta / (1 + iter / 1000);  % Gradually reduce eta to fine-tune the result

        % if diff < tol
        %     break;
        % end

        if diff < tol %&& sum(sign(Phi * x_new) == y) / length(y) > 0.99
            break;  % Converged when norm is small AND sign accuracy is high
        end

        x_prev = x_new;
    end

    % Output
    x_hat = x_new;
    record.iter = iter;
    fprintf('BIHT converged in %d iterations.\n', iter);
end
