package ui

import (
	"html/template"
	"strconv"
)

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

//AddNetIO adds a window to the ui
func (w *Windows) AddNetIO(header, refreshrate, url string, uh Handler, purl string, up Handler, columnsinrow int, beginrow, endofrow bool) {

	i := len(w.windows.DivOutputs)
	j := len(w.windows.HardwareCharts)
	k := len(w.windows.Stats)
	suffix := strconv.Itoa(i + j + k)
	var newrow template.HTML
	var endrow template.HTML

	if beginrow {
		newrow = template.HTML("<div class=\"row\">")
	}
	if endofrow {
		endrow = template.HTML("</div>")
	}
	colwide := colstring(columnsinrow)

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
	w.windows.DivOutputs = append(w.windows.DivOutputs, d)
	w.names = append(w.names, url)
	w.handlers = append(w.handlers, uh)
	if up != nil {
		w.handlers = append(w.handlers, up)
		w.names = append(w.names, purl)
	}

}
