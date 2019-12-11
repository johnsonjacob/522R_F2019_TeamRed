#!/usr/bin/python
import sys
import usb.core
import usb.util
# decimal vendor and product values
#dev = usb.core.find(idVendor=1118, idProduct=1917)
# or, uncomment the next line to search instead by the hexidecimal equivalent
dev = usb.core.find(idVendor=0x0461, idProduct=0x4e22)
# first endpoint
interface = 0
endpoint = dev[0][(0,0)][0]
# if the OS kernel already claimed the device, which is most likely true
# thanks to http://stackoverflow.com/questions/8218683/pyusb-cannot-set-configuration
if dev.is_kernel_driver_active(interface) is True:
  # tell the kernel to detach
  dev.detach_kernel_driver(interface)
  # claim the device
  usb.util.claim_interface(dev, interface)
collected = 0
attempts = 50
data_str = ''
while True:#collected < attempts :
    try:
        data = dev.read(endpoint.bEndpointAddress,endpoint.wMaxPacketSize)
        collected += 1
        data_str = (str(data[1]).rjust(4) + ' ' + str(data[1]).rjust(4) + ' ' + str(data[2]).rjust(4) + ' ' + str(data[3]).rjust(4))
        print(data_str, end='')
        print('\b' * len(data_str), end='', flush = True)
    except usb.core.USBError as e:
        data = None
        if e.args == ('Operation timed out',):
            continue
# release the device
usb.util.release_interface(dev, interface)
# reattach the device to the OS kernel
dev.attach_kernel_driver(interface)
