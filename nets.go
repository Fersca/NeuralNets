package main

import (
	"fmt"
)

func main() {

	fmt.Println("Welcome to Neural Networks in Golang")
	net := &Net{}

	layer1 := &Layer{}
	layer1.addNeurons(10)

	net.addLayer(layer1)
}

////////////////////////////////////////////////////////////////
type Neuron struct {
	inputs []*Weight
}

func (n *Neuron) addWeight(w *Weight) {
	n.inputs = append(n.inputs, w)
}

////////////////////////////////////////////////////////////////
type Net struct {
	layers []*Layer
}

func (n *Net) addLayer(l *Layer) {
	n.layers = append(n.layers, l)
}

////////////////////////////////////////////////////////////////
type Layer struct {
	neurons []*Neuron
}

func (l *Layer) addNeurons(quantity int) {
	for i := 0; i < quantity; i++ {
		neuron := &Neuron{}
		l.neurons = append(l.neurons, neuron)
	}
}

////////////////////////////////////////////////////////////////
type Weight struct {
	origin  Neuron
	destiny Neuron
	weight  int
}
