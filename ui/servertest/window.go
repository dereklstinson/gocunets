package server

type NetworkStats struct {
	columnpercent string
	header        string
	layers        []stats
}

func NewNetworkStats(columnpercent string, header string) *NetworkStats {
	return &NetworkStats{
		columnpercent: columnpercent,
		header:        header,
	}
}

func (n *NetworkStats) AppendLayer(layer stats) {
	n.layers = append(n.layers, layer)
}

type stats struct {
	header string
	stats  []string
}

func (n *NetworkStats) BuildPage() {

}
