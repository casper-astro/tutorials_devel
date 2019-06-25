import casperfpga,time

#Tutorial ADC and DAC interface (Red Pitaya) Python Script to display, read back the ADC snap shot data and the registers

#parameters
#snapshot read length (can be adjusted)
read_length = 600; 


#Connecting to the Red Pitaya
print 'connecting to the Red Pitaya...'
rp=casperfpga.CasperFpga(host='192.168.14.70', port=7147)
print 'done'

#program the Red Pitaya
print 'programming the Red Pitaya...'
rp.transport.upload_to_ram_and_program('tut_adc_dac_2019-06-21_1117.fpg')
print 'done'

#arm the snap shot
print 'arming snapshot block...'
rp.snapshots.adc_in_snap_ss.arm()
print 'done'

#start the snap shot triggering and reset the counters
print 'triggering the snapshot and reset the counters...'
rp.registers.reg_cntrl.write(rst_cntrl = 'pulse')
print 'done'


#grab the snapshots
print 'reading the snapshot...'
adc_in = rp.snapshots.adc_in_snap_ss.read(arm=False)['data'] 
print 'done'

#writing ADC data to disk
print 'writing ADC data to disk...'

# Write each ADC channel's sample data to a file
adc_file = open("adc_data.txt","w")
for array_index in range(0, 1024):
	adc_file.write(str(adc_in['adc_data_ch1'][array_index]))
	adc_file.write("\n")
for array_index in range(0, 1024):	
	adc_file.write(str(adc_in['adc_data_ch2'][array_index]))
	adc_file.write("\n")	
adc_file.close()

print 'done'

#read back the status registers
print 'reading back the status registers...'

print "adc sample count:",rp.registers.adc_sample_cnt.read_uint()
print 'done'
#read back the snapshot captured data
print 'Displaying the snapshot block data...'
print 'ADC SNAPSHOT CAPTURED INPUT'
print '-----------------------------'
print 'Num adc_data_valid adc_data_ch1 adc_data_ch2' 
for i in range(0, read_length):
  print '[%i] %i %i %i'% (i, adc_in['adc_data_valid'][i], adc_in['adc_data_ch1'][i], \
                          adc_in['adc_data_ch2'][i])


print 'done'





