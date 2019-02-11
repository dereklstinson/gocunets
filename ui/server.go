package ui

import (
	"html/template"
	"log"
	"net/http"
	"strconv"

	"github.com/dereklstinson/GoCuNets/ui/gpuperformance"
	"github.com/pkg/browser"
)

/*
Handler interface. This is the cool way to control data without having a global variables.
example:
   (s *slideshow)Handle()func(w http.ResponseWriter, req *http.Request){
		return  func( w http.ResponseWriter, req *http.Request){
		err:=	jpeg.Encode(w, s.img[s.index%len(s.img)], nil)
		if err !=nil{panic(err)}
		s.index++
		}
}
*/
type Handler interface {
	Handle() func(http.ResponseWriter, *http.Request)
}

//Windows are the ui sections of the network
type Windows struct {
	windows          IndexTemplate
	handlers         []Handler
	names            []string
	port             string
	ipaddressandport string
	page             string
}

//IndexTemplate is the indexed template of the networks
type IndexTemplate struct {
	HardwareCharts []HardwareCharts
	DivOutputs     []DivOutputs
	Stats          []Stats
}

//NewWindows creates allows the user to create a bunch of windows to access the neural network
//if "localhost" is passed in ipaddress called then it will open a browser for you automatically.
func NewWindows(ipaddress, port, page string) Windows {
	if ipaddress == "localhost" {
		browser.OpenURL(ipaddress + port + page)
	}
	x := Windows{
		port:             port,
		ipaddressandport: ipaddress + port,
		page:             page,
	}
	hardwarerefresh := 3000
	hardwarerefreshstring := strconv.Itoa(hardwarerefresh)
	mem, t, p := gpuperformance.CreateGPUPerformanceHandlers(hardwarerefresh, 120)

	x.AddHardwareCharts("Gpu Mem Used (MB)", hardwarerefreshstring, "/gpumem/", mem, 3, true, false)
	x.AddHardwareCharts("Temp (C) ", hardwarerefreshstring, "/gputemp/", t, 3, false, false)
	x.AddHardwareCharts("Power (Watts)", hardwarerefreshstring, "/gpupower/", p, 3, false, true)
	return x
}

func colstring(columns int) string {
	switch columns {
	case 10:
		return "column10"
	case 6:
		return "column16_66"
	case 5:
		return "column20"

	case 4:
		return "column25"
	case 3:
		return "column33"

	case 2:
		return "column50"

	case 1:
		return "column100"
	}
	return ""
}

//ServerMain is the test server with just a bunch of images from the harddrive
func ServerMain(windows Windows) {
	//serverlocation := "http://localhost" + testingnewporttest
	indexhandler := func(w http.ResponseWriter, req *http.Request) {
		handleindex(w, windows, webTemp())
	}
	http.HandleFunc("/index", indexhandler)
	for i := range windows.handlers {

		http.HandleFunc(windows.names[i], windows.handlers[i].Handle())
	}
	log.Fatal(http.ListenAndServe(windows.port, nil))

}
func handleindex(w http.ResponseWriter, windows Windows, webpage string) {
	t, err := template.New("webpage").Parse(webpage)

	check(err)

	check(t.Execute(w, windows.windows))
}

//URLs are the urls for the
type URLs struct {
	url []string
}

func holdertemplate2(w http.ResponseWriter, serverlocation string, data []string, webpage string) {
	var divouts []DivOutputs
	t, err := template.New("webpage").Parse(webpage)
	check(err)

	check(t.Execute(w, divouts))

}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
