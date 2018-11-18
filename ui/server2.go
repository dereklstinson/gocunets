package ui

import (
	"html/template"
	"image/jpeg"
	"log"
	"net/http"
	"os"

	"github.com/dereklstinson/GoCuNets/utils/filing"
)

const imagefolderlocation = "/home/derek/Desktop/GANMNIST/"
const webpagelocation = "/home/derek/go/src/github.com/dereklstinson/ui/index.html"
const testingnewport = ":8081"

//ServerMain2 is the main
func ServerMain2() {
	// Hello world, the web server
	url := SetupURL("/loss/", "/inputimage/", "/outputimage/")
	handlers := make([]imageHandler, 0)
	for i := range url.url {
		handlers = append(handlers, makeimagehandler(imagefolderlocation, i*20))
	}
	helloHandler := func(w http.ResponseWriter, req *http.Request) {
		holdertemplate2(w, "nothing", url.url)

		//	w.Header().Set("Cache-Control", "no-store")

	}

	http.HandleFunc("/index", helloHandler)
	for i := range url.url {
		http.HandleFunc(url.url[i], handlers[i].F())

	}

	log.Fatal(http.ListenAndServe(testingnewport, nil))
}

//URLs are the urls for the
type URLs struct {
	url []string
}

type WebPage struct {
	Url1 string
	Url2 string
	Url3 string
}

//SetupURL to stream images
func SetupURL(urls ...string) URLs {
	return URLs{
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
func (h *imageHandler) F() func(w http.ResponseWriter, req *http.Request) {
	return func(w http.ResponseWriter, req *http.Request) {
		file, err := os.Open(h.path[h.index])
		check(err)
		defer file.Close()
		img, err := jpeg.Decode(file)
		check(err)
		//	buffer := new(bytes.Buffer)
		check(jpeg.Encode(w, img, nil))
		if h.index < len(h.path) {
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

func holdertemplate2(w http.ResponseWriter, pagelocation string, data []string) {
	urls := WebPage{
		Url1: "http://localhost:8081/loss/",        //data[0],
		Url2: "http://localhost:8081/inputimage/",  //data[1],
		Url3: "http://localhost:8081/outputimage/", //data[2],
	}
	//t, err := template.ParseFiles(pagelocation)
	t, err := template.New("webpage").Parse(thewebpagetemplate2())
	check(err)

	check(t.Execute(w, urls))

}
func check(err error) {
	if err != nil {
		panic(err)
	}
}
