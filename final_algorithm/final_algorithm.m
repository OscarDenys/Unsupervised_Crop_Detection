clear;
close all;

data_struct = load('example_data');
av_err_diff_matrix = data_struct.('av_err_diff_weed_minus_crop_matrix');
%av_err_diff_matrix = transpose(av_err_diff_matrix);
test_counting = data_struct.('test_counting_matrix');
av_err_diff_matrix = av_err_diff_matrix(1:166, 167:332); %% for thesis figure

%%

threshold = prctile(av_err_diff_matrix, 95, 'all') % 90


figure(1)
crop = av_err_diff_matrix;
crop(av_err_diff_matrix<threshold) = 0;
imagesc(log(abs(crop)))
title('crop')

figure(2)
crop_thresh = crop;
crop_thresh(crop>0) = 1;
imagesc(crop_thresh)
title('crop thresh')


% figure(3)
% weed = av_err_diff_matrix;
% weed(av_err_diff_matrix>threshold) = 0;
% weed = abs(weed);
% imagesc(log(weed))
% title('weed')
% 
% figure(4)
% weed_thresh = weed;
% weed_thresh(weed>0) = 1;
% imagesc(weed_thresh)
% title('weed thresh')
%%
figure(5);
% Display the original gray scale image.
subplot(2, 1, 1);
imshow(crop_thresh*255, []);
%fontSize = 20;
title('Original Grayscale Image', 'Interpreter', 'None');


%figure(99)
%Display the original gray scale image.
subplot(2, 1, 2);
F=fft2(crop);
S=fftshift(log(1+abs(F)));
imshow(S,[]);
title('Spectrum Image', 'Interpreter', 'None');
save('spectrum.mat','F')



%%
figure(6);
%threshold = prctile(F, 99.99, 'all') %99.99749 % 99.995
%threshold = maxk(F, 3)
matrix_without_max = F(abs(F)<max(abs(F(:))));
%threshold = abs(max(matrix_without_max(abs(matrix_without_max)<max(abs(matrix_without_max(:))))))
%threshold = max(max(abs(F)))
threshold = abs(max(max(matrix_without_max)))

F2 = F;
F2(abs(F)<threshold) = 0;

count = F2; 
count_nonzero(count>0) = 1;
count_nonzero = sum(sum(count_nonzero))

S=fftshift(log(1+abs(F2)));
subplot(2, 1, 1);
imshow(S,[]);
title('Reduced Spectrum Image', 'Interpreter', 'None');
save('reduced_spectrum.mat','F2')

noise_red = ifft2(F2);
noise_red = noise_red*100;
subplot(2, 1, 2);
imshow(noise_red, []);
title('Line detection', 'Interpreter', 'None');

%%
figure(7);
binary_mask = noise_red;
threshold = prctile(noise_red, 90, 'all') 
binary_mask(binary_mask<threshold) = 0;
binary_mask(binary_mask>threshold) = 1;
imshow(binary_mask, []);
%title('binary mask', 'Interpreter', 'None');

%%
save('line_detection_mask.mat','binary_mask')

