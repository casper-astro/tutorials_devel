#!/usr/bin/env python
'''
This code demonstrates readount of a SKARAB spectrometer. You need a SKARAB with:
-- a test tone going into the ADC0 input 0
-- adc card in mezzanine slot 2
'''
import casperfpga,casperfpga.snapadc,time,numpy,struct,sys,logging,pylab,matplotlib
import matplotlib.pyplot as plt
import numpy as np

katcp_port=7147
freq_range_mhz = [0,80]


class skarab_adc_rx:

	"""
	Class containing functions to control and configure the SKARAB ADC4x3G-14 mezzanine module containing the ADC32RF80 ADCs.
	"""

	def __init__(self, skarab, mezzanine_site):
		"""
		Initialises skarab_adc_rx object
		
		:param mezzanine_site: Location of mezzanine on SKARAB (0->3)
		:type mezzanine_site: int
		"""
	
		self.mezzanine_site = mezzanine_site
	def ConfigureAdcDdc(self, skarab, adc_input, real_ddc_output_enable):
	
		"""
		Function used to configure the DDCs on the SKARAB ADC4x3G-14 mezzanine module.		
		
		:param skarab: The casperfpga object created for the SKARAB.
		:type skarab: casperfpga
		:param adc_input: The ADC channel to configure (0 -> 3).
		:type adc_input: int
		:param real_ddc_output_enable: Enable/Disable real DDC output values
		:type real_ddc_output_enable: boolean
		"""
	
		ADC = 0
		channel = 'A'
		adc_sample_rate = 3e9
		
		if adc_input == 0:
			ADC = 0
			channel = 'B'
		elif adc_input == 1:
			ADC = 0
			channel = 'A'
		elif adc_input == 2:
			ADC = 1
			channel = 'B'
		else:
			ADC = 1
			channel = 'A'
			
		mezzanine_site = self.mezzanine_site
		i2c_interface = mezzanine_site + 1		
		STM_I2C_DEVICE_ADDRESS = 0x0C # 0x18 shifted down by 1 bit
		adc_sample_rate = 3e9
		decimation_rate = 4	
		DECIMATION_RATE_REG = 0x19
		DDC0_NCO_SETTING_MSB_REG = 0x1A
		DDC0_NCO_SETTING_LSB_REG = 0x1B
		DDC1_NCO_SETTING_MSB_REG = 0x1C
		DDC1_NCO_SETTING_LSB_REG = 0x1D
		ddc0_centre_frequency = 1e9
		ddc1_centre_frequency = 0
		dual_band_mode = False
		DDC_ADC_SELECT = 0x01
		DDC_CHANNEL_SELECT = 0x02
		DUAL_BAND_ENABLE = 0x04
		REAL_DDC_OUTPUT_SELECT = 0x40 # ADD SUPPORT FOR REAL DDC OUTPUT SAMPLES
		SECOND_NYQUIST_ZONE_SELECT = 0x08
		UPDATE_DDC_CHANGE = 0x80
		DDC_CONTROL_REG = 0x1E
	
		# Configure ADC DDC
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DECIMATION_RATE_REG, decimation_rate)

		# Calculate the NCO value
		nco_register_setting = pow(2.0, 16.0) * (ddc0_centre_frequency / adc_sample_rate)
		nco_register_setting = int(nco_register_setting)

		write_byte = (nco_register_setting >> 8) & 0xFF
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC0_NCO_SETTING_MSB_REG, write_byte)

		write_byte = nco_register_setting & 0xFF
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC0_NCO_SETTING_LSB_REG, write_byte)

		# If in dual band mode, calculate the second NCO value
		if dual_band_mode == True:
			nco_register_setting = pow(2.0, 16.0) * (ddc1_centre_frequency / adc_sample_rate)
			nco_register_setting = int(nco_register_setting)

			write_byte = (nco_register_setting >> 8) & 0xFF
			skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC1_NCO_SETTING_MSB_REG, write_byte)

			write_byte = nco_register_setting & 0xFF
			skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC1_NCO_SETTING_LSB_REG, write_byte)

		# Trigger a configuration
		write_byte = 0
		if ADC == 1:
			write_byte = write_byte | DDC_ADC_SELECT

		if channel == 'B':
			write_byte = write_byte | DDC_CHANNEL_SELECT

		if dual_band_mode == True:
			write_byte = write_byte | DUAL_BAND_ENABLE
			
		# 08/08/2018 ADD SUPPORT FOR REAL DDC OUTPUT SAMPLES	
		if real_ddc_output_enable == True:
			write_byte = write_byte | REAL_DDC_OUTPUT_SELECT

		# Determine if in second nyquist zone
		if (ddc0_centre_frequency > (adc_sample_rate / 2)):
			write_byte = write_byte | SECOND_NYQUIST_ZONE_SELECT

		write_byte = write_byte | UPDATE_DDC_CHANGE

		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC_CONTROL_REG, write_byte)

		# Wait for the update to complete
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC_CONTROL_REG)
		read_byte = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)

		timeout = 0
		while (((read_byte[0] & UPDATE_DDC_CHANGE) != 0) and (timeout < 1000)):
			skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, DDC_CONTROL_REG)
			read_byte = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)
			timeout = timeout + 1

		if timeout == 1000:
			print("ERROR: Timeout waiting for configure DDC to complete!")
	def PerformAdcPllSync(self, skarab):
		"""
		Function used to synchronise the ADCs and PLL on the SKARAB ADC4x3G-14 mezzanine module.
		After syncrhonisation is performed, ADC sampling begins.
		
		:param skarab: The casperfpga object created for the SKARAB.
		:type skarab: casperfpga
		"""
		
		SPI_DESTINATION_PLL = 0x0
		SPI_DESTINATION_ADC_0 = 0x2
		SPI_DESTINATION_ADC_1 = 0x3
		ADC_GENERAL_MASTER_PAGE_SEL = 0x0012
		ADC_GENERAL_ADC_PAGE_SEL = 0x0011
		SPI_DESTINATION_DUAL_ADC = 0x8
		ENABLE_PLL_SYNC = 0x01
		ENABLE_ADC_SYNC = 0x02
		HOST_PLL_GPIO_CONTROL_REG = 0x26
		MEZ_CONTROL_REG = 0x01
		STM_I2C_DEVICE_ADDRESS = 0x0C # 0x18 shifted down by 1 bit
		FIRMWARE_VERSION_MAJOR_REG = 0x7E
		FIRMWARE_VERSION_MINOR_REG = 0x7F
		PLL_CHANNEL_OUTPUT_3_CONTROL_HIGH_PERFORMANCE_MODE = 0x00E6
		PLL_CHANNEL_OUTPUT_7_CONTROL_HIGH_PERFORMANCE_MODE = 0x010E
		PLL_CLOCK_OUTPUT_PHASE_STATUS = 0x04
		ADC_MASTER_PDN_SYSREF = 0x0020
		PLL_ALARM_READBACK = 0x007D
		
		mezzanine_site = self.mezzanine_site
		i2c_interface = mezzanine_site + 1
		
		# Get embedded software version
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, FIRMWARE_VERSION_MAJOR_REG)
		major_version = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)
		skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, FIRMWARE_VERSION_MINOR_REG)
		minor_version = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)
		
		# Synchronise PLL and ADC	
		skarab.write_int('pll_sync_start_in', 0)
		skarab.write_int('adc_sync_start_in', 0)
		#skarab.write_int('adc_trig', 0)	
		pll_loss_of_reference = False
		synchronise_mezzanine = [False, False, False, False]
		synchronise_mezzanine[mezzanine_site] = True

		if ((major_version == 1) and (minor_version < 3)):
			# TO DO: Implement LVDS SYSREF
			print("Synchronising PLL with LVDS SYSREF.")

			for mezzanine in range(0, 4):
				print("Checking PLL loss of reference for mezzanine: ", mezzanine)

				if synchronise_mezzanine[mezzanine] == True:

					skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, HOST_PLL_GPIO_CONTROL_REG)
					read_byte = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)

					if ((read_byte[0] & 0x01) == 0x01):
						# PLL reporting loss of reference
						pll_loss_of_reference = True
						print("PLL reporting loss of reference.")
					else:
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_3_CONTROL_HIGH_PERFORMANCE_MODE, 0xD1)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_7_CONTROL_HIGH_PERFORMANCE_MODE, 0xD1)

						# Enable PLL SYNC
						skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, MEZ_CONTROL_REG, ENABLE_PLL_SYNC)

			# Only try to synchronise PLLs if all SKARAB ADC32RF45X2 mezzanines have reference
			if pll_loss_of_reference == False:
				# Synchronise HMC7044 first
				printf("Synchronising PLL.")

				# Trigger a PLL SYNC signal from the MB firmware
				skarab.write_int('pll_sync_start_in', 0)
				skarab.write_int('pll_sync_start_in', 1)

				# Wait for the PLL SYNC to complete
				timeout = 0
				read_reg = skarab.read_int('pll_sync_complete_out')
				while ((read_reg == 0) and (timeout < 100)):
					read_reg = skarab.read_int('pll_sync_complete_out')
					timeout = timeout + 1

				if timeout == 100:
					print("ERROR: Timeout waiting for PLL SYNC to complete!")

				for mezzanine in range(0, 4):
					if synchronise_mezzanine[mezzanine] == True:
						# Disable the PLL SYNC and wait for SYSREF outputs to be in phase
						print("Disabling ADC SYNC on mezzanine: ", mezzanine)

						skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, MEZ_CONTROL_REG, 0x0)

						spi_read_word = self.DirectSpiRead(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_ALARM_READBACK)
						timeout = 0
						while (((spi_read_word & PLL_CLOCK_OUTPUT_PHASE_STATUS) == 0x0) and (timeout < 1000)):
							spi_read_word = self.DirectSpiRead(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_ALARM_READBACK)
							timeout = timeout + 1

						if timeout == 1000:
							print("ERROR: Timeout waiting for the mezzanine PLL outputs to be in phase.")

						# Power up SYSREF input buffer on ADCs
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_GENERAL_ADC_PAGE_SEL, 0x00)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_GENERAL_MASTER_PAGE_SEL, 0x04)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_MASTER_PDN_SYSREF, 0x00)

						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_GENERAL_ADC_PAGE_SEL, 0x00)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_GENERAL_MASTER_PAGE_SEL, 0x04)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_MASTER_PDN_SYSREF, 0x00)

						time.sleep(1)
						
						# Need to disable both at the same time so NCOs have same phase
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_DUAL_ADC, ADC_MASTER_PDN_SYSREF, 0x10)
						
						# Disable SYSREF again
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_3_CONTROL_HIGH_PERFORMANCE_MODE, 0xD0)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_7_CONTROL_HIGH_PERFORMANCE_MODE, 0xD0)

		else:
			print("Synchronising PLL with LVPECL SYSREF.");
			
			# Check first to see if mezzanine has a reference clock
			for mezzanine in range(0, 4):
				print("Checking PLL loss of reference for mezzanine: ", mezzanine)

				if synchronise_mezzanine[mezzanine] == True:

					skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, HOST_PLL_GPIO_CONTROL_REG)
					read_byte = skarab.transport.read_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, 1)

					if ((read_byte[0] & 0x01) == 0x01):
						# PLL reporting loss of reference
						pll_loss_of_reference = True
						print("PLL reporting loss of reference.")
					else:
						# Change the SYNC pin to SYNC source
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_GLOBAL_MODE_AND_ENABLE_CONTROL, 0x41)

						# Change SYSREF to pulse gen mode so don't generate any pulses yet
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_3_CONTROL_FORCE_MUTE, 0x88)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_7_CONTROL_FORCE_MUTE, 0x88)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_3_CONTROL_HIGH_PERFORMANCE_MODE, 0xDD)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_CHANNEL_OUTPUT_7_CONTROL_HIGH_PERFORMANCE_MODE, 0xDD)

						# Enable PLL SYNC
						skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, MEZ_CONTROL_REG, ENABLE_PLL_SYNC)

			# Only try to synchronise PLLs if all SKARAB ADC32RF45X2 mezzanines have reference
			if pll_loss_of_reference == False:
				# Synchronise HMC7044 first
				printf("Synchronising PLL.")

				# Trigger a PLL SYNC signal from the MB firmware
				skarab.write_int('pll_sync_start_in', 0)
				skarab.write_int('pll_sync_start_in', 1)

				# Wait for the PLL SYNC to complete
				timeout = 0
				read_reg = skarab.read_int('pll_sync_complete_out')
				while ((read_reg == 0) and (timeout < 100)):
					read_reg = skarab.read_int('pll_sync_complete_out')
					timeout = timeout + 1

				if timeout == 100:
					print("ERROR: Timeout waiting for PLL SYNC to complete!")

				# Wait for the PLL to report valid SYNC status
				for mezzanine in range(0, 4):
					print("Checking PLL SYNC status for mezzanine: ", mezzanine)
					
					if synchronise_mezzanine[mezzanine] == True:
						spi_read_word = self.DirectSpiRead(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_ALARM_READBACK)
						timeout = 0
						while (((spi_read_word & PLL_CLOCK_OUTPUT_PHASE_STATUS) == 0x0) and (timeout < 1000)):
							spi_read_word = self.DirectSpiRead(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_ALARM_READBACK)
							timeout = timeout + 1

						if timeout == 1000:
							print("ERROR: Timeout waiting for the mezzanine PLL outputs to be in phase.")

				# Synchronise ADCs to SYSREF next
				for mezzanine in range(0, 4):
					print("Using SYSREF to synchronise ADC on mezzanine: ", mezzanine)
					
					if synchronise_mezzanine[mezzanine] == True:
						# Change the SYNC pin to pulse generator
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_PLL, PLL_GLOBAL_MODE_AND_ENABLE_CONTROL, 0x81)

						# Power up SYSREF input buffer on ADCs
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_GENERAL_ADC_PAGE_SEL, 0x00)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_GENERAL_MASTER_PAGE_SEL, 0x04)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_MASTER_PDN_SYSREF, 0x00)

						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_GENERAL_ADC_PAGE_SEL, 0x00)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_GENERAL_MASTER_PAGE_SEL, 0x04)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_MASTER_PDN_SYSREF, 0x00)

				# Trigger a PLL SYNC signal from the MB firmware
				skarab.write_int('pll_sync_start_in', 0)
				skarab.write_int('pll_sync_start_in', 1)

				timeout = 0
				read_reg = skarab.read_int('pll_sync_complete_out')
				while ((read_reg == 0) and (timeout < 100)):
					read_reg = skarab.read_int('pll_sync_complete_out')
					timeout = timeout + 1

				for mezzanine in range(0, 4):
					print("Power down SYSREF buffer for ADC on mezzanine: ", mezzanine)

					if synchronise_mezzanine[mezzanine] == True:
						# Power down SYSREF input buffer on ADCs
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_0, ADC_MASTER_PDN_SYSREF, 0x10)
						self.DirectSpiWrite(skarab, mezzanine_site, SPI_DESTINATION_ADC_1, ADC_MASTER_PDN_SYSREF, 0x10)

		# At this point, all the PLLs across all mezzanine sites should be in sync

		# Enable the ADC SYNC
		for mezzanine in range(0, 4):
			if synchronise_mezzanine[mezzanine] == True:
				print("Enabling ADC SYNC on mezzanine: ", mezzanine)

				skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, MEZ_CONTROL_REG, ENABLE_ADC_SYNC)

		# Trigger a ADC SYNC signal from the MB firmware
		skarab.write_int('adc_sync_start_in', 0)
		skarab.write_int('adc_sync_start_in', 1)

		timeout = 0
		read_reg = skarab.read_int('adc_sync_complete_out')
		while ((read_reg == 0) and (timeout < 1000)):
			read_reg = skarab.read_int('adc_sync_complete_out')
			timeout = timeout + 1

		if timeout == 1000:
			print("ERROR: Timeout waiting for ADC SYNC to complete!")

		# Disable the ADC SYNC
		for mezzanine in range(0, 4):
			if synchronise_mezzanine[mezzanine] == True:
				print("Disabling ADC SYNC on mezzanine: ", mezzanine)

				skarab.transport.write_i2c(i2c_interface, STM_I2C_DEVICE_ADDRESS, MEZ_CONTROL_REG, 0x0)



def get_data():
	#get the data...    
	acc_n = fpga.read_uint('acc_cnt')
	a_1=struct.unpack('>1024Q',fpga.read('mem1',1024*8,0))
	a_2=struct.unpack('>1024Q',fpga.read('mem2',1024*8,0))
	a_3=struct.unpack('>1024Q',fpga.read('mem3',1024*8,0))
	a_4=struct.unpack('>1024Q',fpga.read('mem4',1024*8,0))
	
	interleave_a=[]

	for i in range(1024):
		interleave_a.append(a_1[i])
		interleave_a.append(a_2[i])
		interleave_a.append(a_3[i])
		interleave_a.append(a_4[i])

	return acc_n, numpy.array(interleave_a,dtype=numpy.float) 

def plot_spectrum():
	matplotlib.pyplot.clf()
	acc_n, interleave_a = get_data()
	interleave_a = interleave_a[::-1]
	interleave_a = np.fft.fftshift(interleave_a)
	#interleave_a = 10*numpy.log10(interleave_a)
	print(np.linspace(-4096./2,4096./2-1,4096).shape,interleave_a.shape)
	matplotlib.pylab.plot(np.linspace(-4096./2,4096./2-1,4096)*750./4096.,interleave_a,'b')
	matplotlib.pylab.title('Integration number %i.'%acc_n)
	#matplotlib.pylab.ylabel('Power (dB)')
	matplotlib.pylab.grid()
	matplotlib.pylab.xlabel('Freq (MHz)')
	matplotlib.pylab.xlim(freq_range_mhz[0], freq_range_mhz[1])
	fig.canvas.draw()
	fig.canvas.manager.window.after(100, plot_spectrum)



#START OF MAIN:
if __name__ == '__main__':
	from optparse import OptionParser
	p = OptionParser()
	p.set_usage('spectrometer.py <SKARAB_HOSTNAME_or_IP> [options]')
	p.set_description(__doc__)
	p.add_option('-l', '--acc_len', dest='acc_len', type='int',default=2*(2**28)/2048,
	help='Set the number of vectors to accumulate between dumps. default is 2*(2^28)/2048, or just under 2 seconds.')
	p.add_option('-b', '--fpg', dest='fpgfile',type='str', default='',
	help='Specify the fpg file to load')
	opts, args = p.parse_args(sys.argv[1:])

	if args==[]:
		print 'Please specify a SKARAB board. Run with the -h flag to see all options.\nExiting.'
		exit()
	else:
		skarab = args[0] 
	if opts.fpgfile != '':
		bitstream = opts.fpgfile

try:
	print('Connecting to server %s on port %i... '%(skarab,katcp_port)),
	fpga = casperfpga.CasperFpga(skarab)
	time.sleep(0.2)

	if fpga.is_connected():
		print 'ok\n'
	else:
		print 'ERROR connecting to server %s on port %i.\n'%(skarab,katcp_port)
		exit()

	print '------------------------'
	print 'Programming FPGA with %s...' %bitstream,
	sys.stdout.flush()

	fpga.upload_to_ram_and_program(bitstream)


	# After programming we need to configure the ADC. Assumes ADC card is in mezzanine site 2 and source is on adc_channel 0.
        print 'Attempting to configure and sync ADC...'
	mezzanine_site=2

	# Can set DDC frequency and operating band using	ddc0_centre_frequency = 1e9
	try:
		skarab_adc_rx_obj=skarab_adc_rx(fpga,mezzanine_site)
		skarab_adc_rx_obj.ConfigureAdcDdc(fpga,0,False)
		print 'ADC configured...'
		skarab_adc_rx_obj.PerformAdcPllSync(fpga)
		if fpga.read_int('pll_sync_complete_out'):
			print 'ADC PLL sync is working...'
		else:
			print 'Error in PLL sync...'
			exit()
	except:
		print 'ADC INITIALIZATION FAILED...'
		exit()

	# Set registers.
	print 'Configuring accumulation period...',
	sys.stdout.flush()
	fpga.write_int('acc_len',opts.acc_len)
	print 'done'

	print 'Resetting counters...',
	sys.stdout.flush()
	fpga.write_int('cnt_rst',1) 
	fpga.write_int('cnt_rst',0) 
	print 'done'

	# Sync the ADC
	print 'Syncing the ADC...'
	sys.stdout.flush()



	# Make ADC snapshot file
	fpga.snapshots.adc_snap1_ss.arm()	#pols 0..7
	fpga.snapshots.adc_snap2_ss.arm()	#valid flag and trigger
	fpga.write_int('adc_snap',0)
	fpga.write_int('adc_snap',1)
	fpga.write_int('adc_snap',0)

	adc_snap1_pol0=fpga.snapshots.adc_snap1_ss.read(arm=False)['data']
	adc_snap2_pol0=fpga.snapshots.adc_snap2_ss.read(arm=False)['data']
	adc0i_file = open("adc0i_data.txt","w")
	adc0q_file = open("adc0q_data.txt","w")

	for array_index in range(0, 1024):
		if adc_snap2_pol0['adc0_val']:
			adc0i_file.write(str(adc_snap1_pol0['p0'][array_index]))
			adc0i_file.write("\n")
			adc0i_file.write(str(adc_snap1_pol0['p1'][array_index]))
			adc0i_file.write("\n")
			adc0i_file.write(str(adc_snap1_pol0['p2'][array_index]))
			adc0i_file.write("\n")
			adc0i_file.write(str(adc_snap1_pol0['p3'][array_index]))
			adc0i_file.write("\n")
			adc0q_file.write(str(adc_snap1_pol0['p4'][array_index]))
			adc0q_file.write("\n")
			adc0q_file.write(str(adc_snap1_pol0['p5'][array_index]))
			adc0q_file.write("\n")
			adc0q_file.write(str(adc_snap1_pol0['p6'][array_index]))
			adc0q_file.write("\n")
			adc0q_file.write(str(adc_snap1_pol0['p7'][array_index]))
			adc0q_file.write("\n")

	adc0i_file.close()
	adc0q_file.close()

	#set up the figure with a subplot to be plotted
	fig = matplotlib.pyplot.figure()
	ax = fig.add_subplot(1,1,1)

	# start the process
	fig.canvas.manager.window.after(100, plot_spectrum)
	matplotlib.pyplot.show()
	print 'Plot started.'

		
except KeyboardInterrupt:
    exit()

exit()






