package ui

import (
	"html/template"
	"net/http"
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

const webpagelocation = "/home/derek/go/src/github.com/dereklstinson/GoCuNets/ui/index.html"

//ServerMainTest is the test server with just a bunch of images from the harddrive
func ServerMain(port, pagelocation string) {

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

func holdertemplate2(w http.ResponseWriter, serverlocation string, data []string, webpage string) {
	urls := WebPage{
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
func check(err error) {
	if err != nil {
		panic(err)
	}
}
