package gpuperformance

import (
	"time"

	"github.com/dereklstinson/gocunets/ui/plot"
	"github.com/dereklstinson/gocunets/utils/hwperf"
)

//CreateGPUPerformanceHandlers returns the performance handlers for memory used gpu temp and power.
//Updates the values every ms. Sample amount is the number of samples stored for the plot
func CreateGPUPerformanceHandlers(refreshms, sampleamount int) (*Memory /**CoreClock, *MemClock,*/, *Temp, *Power) {
	buffersize := 3
	temps := make(chan []int, buffersize)
	mem := make(chan []int, buffersize)
	//clockcore := make(chan []int, buffersize)
	pow := make(chan []int, buffersize)
	//	clockmem := make(chan []int, buffersize)
	devs, err := hwperf.GetDevices()
	if err != nil {
		panic(err)
	}
	refreshseconds := refreshms / 1000
	go runchannel(devs, refreshms, mem /* clockmem, clockcore,*/, temps, pow)
	m /*cc, mc,*/, t, p := makeMemory(mem, len(devs), sampleamount, refreshseconds),
		//	makeCoreClock(clockcore, len(devs), Lengthoftime),
		//makeMemClock(clockmem, len(devs), Lengthoftime),
		makeTemp(temps, len(devs), sampleamount, refreshseconds),
		makePower(pow, len(devs), sampleamount, refreshseconds)

	return m /*cc, mc,*/, t, p
}

func runchannel(device []*hwperf.Device, refreshms int, mem /*clockcore,clockmem,*/, temp, power chan<- []int) {
	ticker := time.NewTicker(time.Duration(refreshms) * time.Millisecond)
	dl := len(device)
	//devcoreclocks := make([]int, dl)
	//devmemclocks := make([]int, dl)
	memused := make([]int, dl)
	//memfree := make([]int, dl)
	powers := make([]int, dl)
	temps := make([]int, dl)

	for {
		<-ticker.C
		for i := range device {
			device[i].RefreshStatus()
			//	dc, dm := device[i].Clocks()
			//	devcoreclocks[i], devmemclocks[i] = int(dc), int(dm)
			temps[i] = int(device[i].Temp())
			powers[i] = int(device[i].Power())
			usedm, _ := device[i].Memory()
			memused[i] = int(usedm)

		}
		mem <- memused
		temp <- temps
		power <- powers
		//	clockcore <- devcoreclocks
		//	clockmem <- devmemclocks

	}

}
func placeandshiftback(a plot.LabeledData, in int) {
	b := 0.0
	input := float64(in)
	var backwards int
	for i := a.Data.Len() - 1; i >= 0; i-- {
		b = a.Data[i].Y
		a.Data[i].Y = input
		a.Data[i].X = float64(backwards)
		input = b
		backwards--
	}
}
