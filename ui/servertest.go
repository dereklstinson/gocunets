package ui

import (
	"html/template"
	"image/jpeg"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/dereklstinson/GoCuNets/utils/filing"
)

const imagefolderlocationtest = "/home/derek/Desktop/mpii/images/"
const webpagelocationtest = "/home/derek/go/src/github.com/dereklstinson/GoCuNets/ui/index_test.html"
const testingnewporttest = ":8081"

//ServerMainTest is the test server with just a bunch of images from the harddrive
func ServerMainTest() {
	pagebytes, err := ioutil.ReadFile(webpagelocationtest)
	check(err)
	url := []string{"/loss/", "/inputimage/", "/outputimage/", "/extra1/", "/extra2/", "/extra3/"}
	headers := []string{"LALALAL", "BLABLALBA", "CHAHAHAH", "LOINWEFG", "jfgasdkl", "CRAZY"}
	paragraphs := []string{"ppppLALALAL", "pppppBLABLALBA", "ppppppCHAHAHAH", "pppppLOINWEFG", "pppppjfgasdkl", "pppppCRAZY"}
	refreshrate := []string{"10", "20", "100", "200", "1000", "2000"}
	wpt2 := MakeWebPageTest2(testingnewporttest, 3, headers, paragraphs, refreshrate, url)
	handlers := make([]imageHandler, 0)
	for i := range url {
		handlers = append(handlers, makeimagehandler(imagefolderlocationtest, i*20))
	}
	//serverlocation := "http://localhost" + testingnewporttest
	helloHandler := func(w http.ResponseWriter, req *http.Request) {
		webpagesectiontemplate2(w, wpt2, string(pagebytes))

		//	w.Header().Set("Cache-Control", "no-store")

	}

	http.HandleFunc("/index", helloHandler)
	for i := range url {
		http.HandleFunc(url[i], handlers[i].Handle())

	}

	log.Fatal(http.ListenAndServe(testingnewporttest, nil))
}

type DivOutputsTest struct {
	Header    string
	URL       string
	ID        string
	Paragraph string
	NewRow    template.HTML
	EndRow    template.HTML
	Func      template.JS
	MyVar     template.JS
	Rate      template.JS
}

func MakeWebPageTest2(port string, columns int, headers, paragraphs, rate, url []string) []DivOutputsTest {
	x := make([]DivOutputsTest, len(url))
	newrow := template.HTML("<div class=\"row\">")
	endrow := template.HTML("</div>")
	for i := range x {
		suffix := strconv.Itoa(i)
		if i%columns == 0 {
			x[i].NewRow = newrow
		}
		if i%columns == columns-1 {
			x[i].EndRow = endrow
		}
		x[i].Rate = template.JS(rate[i])
		//	x[i].Refresh = template.JS("refvar" + suffix)
		x[i].URL = "http://localhost" + port + url[i]
		x[i].Paragraph = paragraphs[i]
		x[i].Header = headers[i]
		x[i].ID = "image" + suffix
		x[i].Func = template.JS("imagefunc" + suffix)
		x[i].MyVar = template.JS("myvar" + suffix)

	}
	//if (len(x)-1)%columns != columns-1{
	x[len(x)-1].EndRow = endrow
	//	}
	return x
}

type imageHandler struct {
	path  []string
	index int
}

func makeimagehandler(folder string, startindex int) imageHandler {

	return imageHandler{
		index: startindex,
		path:  filing.GetFilePaths(folder),
	}
}
func (h *imageHandler) Handle() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		file, err := os.Open(h.path[h.index])
		check(err)
		defer file.Close()
		img, err := jpeg.Decode(file)
		check(err)
		//	buffer := new(bytes.Buffer)
		check(jpeg.Encode(w, img, nil))
		if h.index < len(h.path)-1 {
			h.index++
		} else {
			h.index = 0
		}
		/*
			key := "loss"
			e := `"` + key + `"`
			w.Header().Set("Etag", e)
			w.Header().Set("Cache-Control", "no-store")
			if match := req.Header.Get("If-None-Match"); match != "" {
				if strings.Contains(match, e) {
					w.WriteHeader(http.StatusNotModified)
					return
				}
			}*/

	}

}
func webpagesectiontemplate2(w http.ResponseWriter, sections []DivOutputsTest, webpage string) {

	//t, err := template.ParseFiles(pagelocation)
	//t, err := template.New("index").ParseFiles(webpagelocation)
	t, err := template.New("webpage").Parse(webpage)
	check(err)

	check(t.Execute(w, sections))

}
