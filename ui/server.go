package ui

import (
	"html/template"
	"log"
	"net/http"
	"strconv"

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

type Windows struct {
	windows          []DivOutputs
	handlers         []Handler
	names            []string
	columnsperrow    []int
	currentrow       int
	currentcol       int
	port             string
	ipaddressandport string
	page             string
}

//NewWindows creates allows the user to create a bunch of windows to access the neural network
func NewWindows(columnsperrow []int, ipaddress, port, page string) Windows {
	browser.OpenURL(ipaddress + port + page)
	return Windows{
		port:             port,
		columnsperrow:    columnsperrow,
		ipaddressandport: ipaddress + port,
		page:             page,
	}
}

//func (w *Windows) AddWindowV2(header, refreshrate, url string, image Handler, paragraph string, para Handler)

//AddWindow adds a window to the ui
func (w *Windows) AddWindow(header, refreshrate, url string, uh Handler, purl string, up Handler) {

	i := len(w.windows)
	suffix := strconv.Itoa(i)
	var newrow template.HTML
	var endrow template.HTML

	if i%w.columnsperrow[w.currentrow] == 0 {
		newrow = template.HTML("<div class=\"row\">")
	}
	if i%w.columnsperrow[w.currentrow] == w.columnsperrow[w.currentrow]-1 {
		endrow = template.HTML("</div>")
	}

	//	x[i].Refresh = template.JS("refvar" + suffix)
	colwide := colstring(w.columnsperrow[w.currentrow])
	d := DivOutputs{
		ColWid: colwide,
		NewRow: newrow,
		EndRow: endrow,
		Header: header,
		PURL:   w.ipaddressandport + purl,
		Name:   url,
		Rate:   template.JS(refreshrate),
		URL:    w.ipaddressandport + url,
		PID:    "para" + suffix,
		ID:     "image" + suffix,
		Func:   template.JS("imagefunc" + suffix),
		MyVar:  template.JS("myvar" + suffix),
	}
	w.names = append(w.names, url)
	w.names = append(w.names, purl)
	w.windows = append(w.windows, d)
	w.handlers = append(w.handlers, uh)
	w.handlers = append(w.handlers, up)
	w.currentcol++
	if !(w.currentcol < w.columnsperrow[w.currentrow]) {
		w.currentcol = 0
		if w.currentrow < len(w.columnsperrow)-1 {
			w.currentrow++
		}
	}
}

func (w *Windows) endlastrow() {
	i := len(w.windows) - 1
	w.windows[i].EndRow = template.HTML("</div>")
}
func colstring(columns int) string {
	switch columns {
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

const webpagelocation = "/home/derek/go/src/github.com/dereklstinson/GoCuNets/ui/index.html"

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

//DivOutputs is a struct that helps build a dynamic output for the ui
type DivOutputs struct {
	Header string
	URL    string
	PURL   string
	ID     string
	PID    string
	Name   string
	ColWid string
	NewRow template.HTML
	EndRow template.HTML
	Func   template.JS
	MyVar  template.JS
	Rate   template.JS
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
