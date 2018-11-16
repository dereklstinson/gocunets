package ui

import (
	"bytes"
	"encoding/base64"
	"html/template"
	"image/jpeg"
	"log"
	"net/http"
	"os"
)

const ipaddressserverlocation = ":8080"

//ServerMain is the main
func ServerMain() {
	// Hello world, the web server

	helloHandler := func(w http.ResponseWriter, req *http.Request) {
		holdertemplate(w)
	}

	http.HandleFunc("/hello", helloHandler)
	log.Fatal(http.ListenAndServe(ipaddressserverlocation, nil))
}

const testimagelocation = "/home/derek/Desktop/mpii/images/000013469.jpg"

//First test Ui mainpage
type UImainpage struct {
	//Loss
	LossHeader string
	LSrc       string
	LossInfo   string
	LWidth     int
	LHeight    int
	//Name
	Name   string
	Src    string
	Width  int
	Height int
	Info   string
	//Name1
	Name1   string
	Src1    string
	Width1  int
	Height1 int
	Info1   string
}

func builduimainpagetest() UImainpage {
	var some UImainpage
	some.LossHeader = "ITS A LOSS"
	some.LSrc = getjpeg()
	some.LossInfo = "This isn't loss"
	some.Name = "TESTING"
	some.Src = getjpeg()
	some.Info = "same image"
	some.Name1 = "Testing1"
	some.Src1 = getjpeg()
	some.Info1 = "same image1"
	return some
}
func getjpeg() string {
	file, err := os.Open(testimagelocation)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	img, err := jpeg.Decode(file)
	if err != nil {
		panic(err)
	}
	buffer := new(bytes.Buffer)
	err = jpeg.Encode(buffer, img, nil)
	if err != nil {
		panic(err)
	}

	return base64.StdEncoding.EncodeToString(buffer.Bytes())
}

func holdertemplate(w http.ResponseWriter) {
	check := func(err error) {
		if err != nil {
			log.Fatal(err)
		}
	}
	data := builduimainpagetest()
	t, err := template.New("webpage").Parse(thewebpagetemplate())
	check(err)

	err = t.Execute(w, data)
	check(err)

}
