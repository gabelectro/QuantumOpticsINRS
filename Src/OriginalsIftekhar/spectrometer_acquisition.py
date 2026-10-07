from pyAndorSDK2 import atmcd, atmcd_codes, atmcd_errors
from matplotlib import pyplot as plt
import time
sdk = atmcd()  # Load the atmcd library
codes = atmcd_codes

ret = sdk.Initialize("")  # Initialize camera
print("Function Initialize returned {}".format(ret))

(ret, iSerialNumber) = sdk.GetCameraSerialNumber()
print("Function GetCameraSerialNumber returned {} Serial No: {}".format(
    ret, iSerialNumber))

ret = sdk.CoolerON()
set_temp = -85
ret = sdk.SetTemperature(set_temp);
(ret, temperature) = sdk.GetTemperature()
while temperature > -30:
    (ret, temperature) = sdk.GetTemperature()
    time.sleep(5)
    print("Temperature is {}".format(temperature))SCAN)
print("Function SetAcquisitionMode returned {}".format(ret))
ret = sdk.SetReadMode(codes.Read_Mode.FULL_VERTICAL_BINNING)
print("Function SetReadMode returned {}".format(ret))
ret = sdk.SetExposureTime(0.5)
print("Function SetExposureTime returned {}".format(ret))
ret = sdk.SetTriggerMode(codes.Trigger_Mode.INTERNAL)
print("Function SetTriggerMode returned {}".format(ret))
ret = sdk.StartAcquisition()
print("Function StartAcquisition returned {}".format(ret))
while (1):
    ret = sdk.WaitForAcquisition()
    print("Function WaitForAcquisition returned {}".format(ret))
    if atmcd_errors.Error_Codes.DRV_SUCCESS == ret:
        break

(ret, arr) = sdk.GetAcquiredData(1024)
print("Function GetAcquiredData returned {}".format(ret))
plt.plot(arr)
#ret = sdk.ShutDown()