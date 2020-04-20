package gocunets

import "testing"

func TestSetPeerAccess(t *testing.T) {
	devs, err := GetDeviceList()
	if err != nil {
		t.Error(err)
	}
	nconnections, err := SetPeerAccess(devs)
	if err != nil {
		t.Error(err)
	}
	t.Error(nconnections)
}
