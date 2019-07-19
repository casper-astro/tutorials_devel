adc_file = fopen('adc_data.txt','r');
adc_all = fscanf(adc_file, '%i');
adc_data_ch1 = adc_all(1:1024);
adc_data_ch2 = adc_all(1025:2048);

figure(1)
plot(adc_data_ch1)
figure(2)
plot(adc_data_ch2)



