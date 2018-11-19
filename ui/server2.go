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
	url := SetupURL("/loss/", "/inputimage/", "/outputimage/", "/extra1/", "/extra2/", "/extra3/")
	handlers := make([]imageHandler, 0)
	for i := range url.url {
		handlers = append(handlers, makeimagehandler(imagefolderlocation, i*20))
	}
	serverlocation := "http://localhost" + testingnewport
	helloHandler := func(w http.ResponseWriter, req *http.Request) {
		holdertemplate2(w, serverlocation, url.url)

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

//WebPage is the webpage stuff
type WebPage struct {
	Url1 string
	Url2 string
	Url3 string
	Url4 string
	Url5 string
	Url6 string
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

func holdertemplate2(w http.ResponseWriter, serverlocation string, data []string) {
	urls := WebPage{
		Url1: serverlocation + data[0],
		Url2: serverlocation + data[1],
		Url3: serverlocation + data[2],
		Url4: serverlocation + data[3],
		Url5: serverlocation + data[4],
		Url6: serverlocation + data[5],
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
