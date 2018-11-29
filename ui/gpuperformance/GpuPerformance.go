package gpuperformance

import (
	"time"

	"github.com/dereklstinson/GoCuNets/ui/plot"
	"github.com/dereklstinson/GoCuNets/utils/hwperf"
)

//CreateGPUPerformanceHandlers
func CreateGPUPerformanceHandlers(refreshms, Lengthoftime int) (*Memory, *CoreClock, *MemClock, *Temp, *Power) {
	buffersize := 3
	temps := make(chan []int, buffersize)
	mem := make(chan []int, buffersize)
	clockcore := make(chan []int, buffersize)
	pow := make(chan []int, buffersize)
	clockmem := make(chan []int, buffersize)
	devs, err := hwperf.GetDevices()
	if err != nil {
		panic(err)
	}
	go runchannel(devs, refreshms, mem, clockmem, clockcore, temps, pow)
	m, cc, mc, t, p := makeMemory(mem, len(devs), Lengthoftime),
		makeCoreClock(clockcore, len(devs), Lengthoftime),
		makeMemClock(clockmem, len(devs), Lengthoftime),
		makeTemp(temps, len(devs), Lengthoftime),
		makePower(pow, len(devs), Lengthoftime)

	return m, cc, mc, t, p
}

func runchannel(device []*hwperf.Device, refreshms int, mem, clockmem, clockcore, temp, power chan<- []int) {
	ticker := time.NewTicker(time.Duration(refreshms) * time.Millisecond)
	dl := len(device)
	devcoreclocks := make([]int, dl)
	devmemclocks := make([]int, dl)
	memused := make([]int, dl)
	//memfree := make([]int, dl)
	powers := make([]int, dl)
	temps := make([]int, dl)

	for {
		<-ticker.C
		for i := range device {
			device[i].RefreshStatus()
			dc, dm := device[i].Clocks()
			devcoreclocks[i], devmemclocks[i] = int(dc), int(dm)
			temps[i] = int(device[i].Temp())
			powers[i] = int(device[i].Power())
			usedm, _ := device[i].Memory()
			memused[i] = int(usedm)

		}
		mem <- memused
		temp <- temps
		power <- powers
		clockcore <- devcoreclocks
		clockmem <- devmemclocks

	}

}
func placeandshiftback(a plot.LabeledData, in int) {
	b := 0.0
	input := float64(in)

	for i := a.Data.Len() - 1; i >= 0; i-- {
		b = a.Data[i].Y
		a.Data[i].Y = input
		input = b
	}
}
