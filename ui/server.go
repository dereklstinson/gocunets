package ui

import (
	"html/template"
	"log"
	"net/http"
	"strconv"
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
	columns          int
	ipaddressandport string
	page             string
}

func NewWindows(columns int, ipaddresswithport, page string) Windows {
	return Windows{
		columns:          columns,
		ipaddressandport: ipaddresswithport,
		page:             page,
	}
}
func (w *Windows) AddWindow(header, paragraph, refreshrate, url string, h Handler) {
	i := len(w.windows) - 1
	suffix := strconv.Itoa(i)
	var newrow template.HTML
	var endrow template.HTML

	if i%w.columns == 0 {
		newrow = template.HTML("<div class=\"row\">")
	}
	if i%w.columns == w.columns-1 {
		endrow = template.HTML("</div>")
	}

	//	x[i].Refresh = template.JS("refvar" + suffix)

	d := DivOutputs{
		NewRow:    newrow,
		EndRow:    endrow,
		Header:    header,
		Paragraph: paragraph,
		Name:      url,
		Rate:      template.JS(refreshrate),
		URL:       w.ipaddressandport + url,
		ID:        "image" + suffix,
		Func:      template.JS("imagefunc" + suffix),
		MyVar:     template.JS("myvar" + suffix),
	}
	w.windows = append(w.windows, d)
	w.handlers = append(w.handlers, h)
}
func (w *Windows) endlastrow() {
	i := len(w.windows) - 1
	w.windows[i].EndRow = template.HTML("</div>")
}

const webpagelocation = "/home/derek/go/src/github.com/dereklstinson/GoCuNets/ui/index.html"

//ServerMain is the test server with just a bunch of images from the harddrive
func ServerMain(windows Windows) {
	//serverlocation := "http://localhost" + testingnewporttest
	indexhandler := func(w http.ResponseWriter, req *http.Request) {
		handleindex(w, windows, webTemp())
	}
	http.HandleFunc("/index", indexhandler)
	for i := range windows.windows {
		http.HandleFunc(windows.windows[i].Name, windows.handlers[i].Handle())
	}
	log.Fatal(http.ListenAndServe(testingnewporttest, nil))

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
	Header    string
	URL       string
	ID        string
	Paragraph string
	Name      string
	NewRow    template.HTML
	EndRow    template.HTML
	Func      template.JS
	MyVar     template.JS
	Rate      template.JS
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
