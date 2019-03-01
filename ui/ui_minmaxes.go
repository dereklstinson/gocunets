package ui

import (
	"html/template"
	"strconv"
)

//Stats are the minmaxes of the network layers or hidden io
type Stats struct {
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

//AddStats adds a window to the ui
func (w *Windows) AddStats(header, refreshrate, purl string, up Handler, columnsinrow int, beginrow, endofrow bool) {

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

	d := Stats{
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
	w.windows.Stats = append(w.windows.Stats, d)
	w.handlers = append(w.handlers, up)
	w.names = append(w.names, purl)

}
