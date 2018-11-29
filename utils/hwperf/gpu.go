package hwperf

import (
	"github.com/dereklstinson/gpu-monitoring-tools/bindings/go/nvml"
)

//Device contains the method of which we can get device info
type Device struct {
	d *nvml.Device
	s *nvml.DeviceStatus
}

//GetDevices returns a slice of devices
func GetDevices() ([]*Device, error) {
	amount, err := nvml.GetDeviceCount()
	if err != nil {
		return nil, err
	}
	devices := make([]*Device, amount)
	for i := uint(0); i < amount; i++ {
		devices[i].d, err = nvml.NewDevice(i)
		if err != nil {
			return nil, err
		}
		devices[i].s, err = devices[i].d.Status()
		if err != nil {
			return nil, err
		}
	}
	return devices, nil
}

//RefreshStatus refreshes the device status
func (d *Device) RefreshStatus() (err error) {
	d.s, err = d.d.Status()
	return err
}

//Clocks returns the Core and Memory clock speed in MHz
func (d *Device) Clocks() (Cores, Mem uint) {
	return *d.s.Clocks.Cores, *d.s.Clocks.Memory
}

//Memory returns the amount used and free on device
func (d *Device) Memory() (Used, Free uint) {
	u := *d.s.Memory.Global.Used
	f := *d.s.Memory.Global.Free
	return uint(u), uint(f)
}

//Temp returns the device temp
func (d *Device) Temp() (Celcius uint) {
	return *d.s.Temperature
}

//Power returns the device power
func (d *Device) Power() (Watts uint) {
	return *d.s.Power
}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
