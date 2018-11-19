package ui

import (
	"html/template"
	"image/jpeg"
	"io/ioutil"
	"log"
	"net/http"
	"os"

	"github.com/dereklstinson/GoCuNets/utils/filing"
)

const imagefolderlocationtest = "/home/derek/Desktop/mpii/images/"
const webpagelocationtest = "/home/derek/go/src/github.com/dereklstinson/GoCuNets/ui/index.html"
const testingnewporttest = ":8081"

//ServerMainTest is the test server with just a bunch of images from the harddrive
func ServerMainTest() {
	pagebytes, err := ioutil.ReadFile(webpagelocationtest)
	check(err)
	url := SetupURL("/loss/", "/inputimage/", "/outputimage/", "/extra1/", "/extra2/", "/extra3/")
	handlers := make([]imageHandler, 0)
	for i := range url.url {
		handlers = append(handlers, makeimagehandler(imagefolderlocationtest, i*20))
	}
	serverlocation := "http://localhost" + testingnewporttest
	helloHandler := func(w http.ResponseWriter, req *http.Request) {
		holdertemplate2test(w, serverlocation, url.url, string(pagebytes))

		//	w.Header().Set("Cache-Control", "no-store")

	}

	http.HandleFunc("/index", helloHandler)
	for i := range url.url {
		http.HandleFunc(url.url[i], handlers[i].Handle())

	}

	log.Fatal(http.ListenAndServe(testingnewporttest, nil))
}

//URLs are the urls for the
type URLsTest struct {
	url []string
}

//WebPage is the webpage stuff
type WebPageTest struct {
	Url1 string
	Url2 string
	Url3 string
	Url4 string
	Url5 string
	Url6 string
}

//SetupURLTest to stream images
func SetupURLTest(urls ...string) URLsTest {
	return URLsTest{
		url: urls,
	}
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

func holdertemplate2test(w http.ResponseWriter, serverlocation string, data []string, webpage string) {
	urls := WebPageTest{
		Url1: serverlocation + data[0],
		Url2: serverlocation + data[1],
		Url3: serverlocation + data[2],
		Url4: serverlocation + data[3],
		Url5: serverlocation + data[4],
		Url6: serverlocation + data[5],
	}
	//t, err := template.ParseFiles(pagelocation)
	//t, err := template.New("index").ParseFiles(webpagelocation)
	t, err := template.New("webpage").Parse(webpage)
	check(err)

	check(t.Execute(w, urls))

}
