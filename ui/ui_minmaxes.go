package ui

import (
	"html/template"
	"strconv"
)

//MinMaxes are the minmaxes of the network layers or hidden io
type MinMaxes struct {
	Header string
	PURL   string
	PID    string
	Name   string
	ColWid string
	NewRow template.HTML
	EndRow template.HTML
	Func   template.JS
	MyVar  template.JS
	Rate   template.JS
}

//AddMinMax adds a window to the ui
func (w *Windows) AddMinMax(header, refreshrate, purl string, up Handler, columnsinrow int, beginrow, endofrow bool) {

	i := len(w.windows.DivOutputs)
	j := len(w.windows.HardwareCharts)
	k := len(w.windows.MinMaxes)
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

	d := MinMaxes{
		ColWid: colwide,
		NewRow: newrow,
		EndRow: endrow,
		Header: header,
		PURL:   w.ipaddressandport + purl,
		Name:   purl,
		Rate:   template.JS(refreshrate),

		PID: "para" + suffix,

		Func:  template.JS("imagefunc" + suffix),
		MyVar: template.JS("myvar" + suffix),
	}
	w.windows.MinMaxes = append(w.windows.MinMaxes, d)
	w.handlers = append(w.handlers, up)
	w.names = append(w.names, purl)

}
func NetworkStatHelper()
