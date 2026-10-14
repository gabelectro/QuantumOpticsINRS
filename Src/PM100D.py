import pyvisa
rm = pyvisa.ResourceManager()
try:
    pm100d = rm.open_resource('USB0::0x1313::0x8078::P0042558::0::INSTR')  #PM100D's VISA address
    print(pm100d.query("SYST:SENS:IDN?"))
    #pm100d.write('*RST')  # Reset the instrument
    pm100d.write('SENS:FUNC "POW"')  # Set to measure power
    pm100d.write('SENS:POW:UNIT W')  # Set power unit to Watts
except pyvisa.VisaIOError as e:
    print(f"Error occurred while communicating with the instrument: {e}")
