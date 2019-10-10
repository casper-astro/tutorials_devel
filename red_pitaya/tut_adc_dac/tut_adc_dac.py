import casperfpga, time, sys

from optparse import OptionParser

default_fpg = 'tut_adc_dac/outputs/tut_adc_dac_2019-08-05_1755.fpg'

p = OptionParser()
p.set_usage('tut_adc_dac.py <BOARD_HOSTNAME_or_IP> [options]')
p.set_description(__doc__)
p.add_option('-b', '--fpgfile', dest='fpg', type='str', default=default_fpg,
    help='Specify the fpg file to load')  
p.add_option('-p', '--plot', dest='plot', action='store_true', default=False,
    help='Plot ADC outputs. This requires the python matplotlib library')
opts, args = p.parse_args(sys.argv[1:])

if args==[]:
    print 'Please specify a board hostname or IP address. \nExiting.'
    sys.exit()
else:
    host = args[0]
if opts.fpg != '':
    fpgfile = opts.fpg

#Tutorial ADC and DAC interface (Red Pitaya) Python Script to display, read back the ADC snap shot data and the registers

#parameters
#snapshot read length (can be adjusted)
read_length = 600; 


#Connecting to the Red Pitaya
print 'connecting to the Red Pitaya...'
rp=casperfpga.CasperFpga(host=host, port=7147)
print 'done'

#program the Red Pitaya
print 'programming the Red Pitaya...'
rp.upload_to_ram_and_program(opts.fpg)
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
with open("adc_data.txt","w") as adc_file:
    for array_index in range(0, 1024):
        adc_file.write(str(adc_in['adc_data_ch1'][array_index]))
        adc_file.write("\n")
    for array_index in range(0, 1024):  
        adc_file.write(str(adc_in['adc_data_ch2'][array_index]))
        adc_file.write("\n")    
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

if opts.plot:
    print 'Plotting ADC captures'
    from matplotlib import pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(adc_in['adc_data_ch1'])
    plt.subplot(2,1,2)
    plt.plot(adc_in['adc_data_ch2'])
    plt.show()
    print 'done'
