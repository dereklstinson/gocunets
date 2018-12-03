package ui

import (
	"html/template"
	"strconv"
)

type HardwareCharts struct {
	Header string
	URL    string
	ID     string
	Name   string
	ColWid string
	NewRow template.HTML
	EndRow template.HTML
	Func   template.JS
	MyVar  template.JS
	Rate   template.JS
}

//AddHardwareCharts adds the hardware charts to the ui for mem temp and power
func (w *Windows) AddHardwareCharts(header, refreshrate, url string, handle Handler, columnsinrow int, beginrow bool, endofrow bool) {
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
	d := HardwareCharts{
		ColWid: colwide,
		NewRow: newrow,
		EndRow: endrow,
		Header: header,
		Name:   url,
		Rate:   template.JS(refreshrate),
		URL:    w.ipaddressandport + url,
		ID:     "image" + suffix,
		Func:   template.JS("imagefunc" + suffix),
		MyVar:  template.JS("myvar" + suffix),
	}
	w.names = append(w.names, url)
	w.handlers = append(w.handlers, handle)
	w.windows.HardwareCharts = append(w.windows.HardwareCharts, d)
}
