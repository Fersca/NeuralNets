package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"time"
)

func main() {

	rand.Seed(time.Now().UTC().UnixNano())

	fmt.Println("Welcome to Neural Networks in Golang")

	//generación
	generation := []*Net{}

	//Crea 10 redes
	for i := 0; i < 10; i++ {

		net := &Net{}

		inputLayer := &Layer{name: "Input Layer"}
		inputLayer.addNeurons(2)

		hiddenLayer := &Layer{name: "Hidden Layer"}
		hiddenLayer.addNeurons(3)

		outputLayer := &Layer{name: "Output Layer"}
		outputLayer.addNeurons(1)

		inputLayer.connectToLayer(hiddenLayer)
		hiddenLayer.connectToLayer(outputLayer)

		net.addLayer(inputLayer)
		net.addLayer(hiddenLayer)
		net.addLayer(outputLayer)

		net.randomizeWeights()

		generation = append(generation, net)

	}

	testCases := []*TestCase{}

	test1 := &TestCase{input: []float64{1, 1}, output: []float64{0}}
	test2 := &TestCase{input: []float64{0, 0}, output: []float64{0}}
	test3 := &TestCase{input: []float64{1, 0}, output: []float64{1}}
	test4 := &TestCase{input: []float64{0, 1}, output: []float64{1}}

	testCases = append(testCases, test1)
	testCases = append(testCases, test2)
	testCases = append(testCases, test3)
	testCases = append(testCases, test4)

	calculateDiffinGeneration(generation, testCases)

	for _, n := range generation {
		fmt.Println("diff: ", n.diff)
	}

	//borra 2
	generation = deleteWorstElements(generation)
	generation = deleteWorstElements(generation)

	//reproduce 2

	fmt.Println("Cantidad: ", len(generation))

	for _, n := range generation {
		fmt.Println("diff: ", n.diff)
	}

	//net.printNetwork()

}

func deleteWorstElements(generation []*Net) []*Net {

	var maxDiff float64
	var worstElementid int

	for id, n := range generation {
		if n.diff > maxDiff {
			maxDiff = n.diff
			worstElementid = id
		}
	}

	cleanGeneration := []*Net{}

	for id, n := range generation {
		if id != worstElementid {
			cleanGeneration = append(cleanGeneration, n)
		}
	}

	return cleanGeneration

}

func calculateDiffinGeneration(generation []*Net, testCases []*TestCase) {

	//valido todas las redes de la generación
	for _, n := range generation {

		//calculo todos los test cases
		var totalDiff float64
		for _, t := range testCases {

			output := n.processInput(t.input)
			//arreglar esto para que sea agnóstico a la cantidad de outputs
			diff := output[0] - t.output[0]
			diff = diff * diff // lo elevo al cuadrado para sumar errores
			totalDiff = totalDiff + diff
		}

		//me guardo la direrencia total en la red
		n.diff = totalDiff

	}

}

type TestCase struct {
	input  []float64
	output []float64
}

////////////////////////////////////////////////////////////////
type Neuron struct {
	connections []*Connection
	name        string
	output      float64
}

func (n *Neuron) addWeight(w *Connection) {
	n.connections = append(n.connections, w)
}

func (n *Neuron) calculateOutput() {
	//Cada neurona debe ver sus inputs, tomar los outputs de cada uno y multiplicarlo por
	//el peso para generar el polinomio

	//Si es la capa de entrada no tiene inputs, y ya tiene los valors seteados como outputs
	if n.connections != nil {
		for _, connection := range n.connections {
			n.output = n.output + connection.origin.output*connection.weight
		}
	}

	//Lo pasa a travéz de la función sigmoidal para normalizarlos
	n.output = 1.0 / (1.0 + math.Exp(-n.output))

}

////////////////////////////////////////////////////////////////
type Net struct {
	layers []*Layer
	diff   float64
}

func (n *Net) addLayer(l *Layer) {
	n.layers = append(n.layers, l)
}

func (n *Net) printNetwork() {

	for _, layer := range n.layers {
		fmt.Println("Layer: ", layer.name)
		fmt.Println("----------------------------")
		for _, neuron := range layer.neurons {
			fmt.Println("Neuron: ", neuron.name, "output: ", neuron.output)
			fmt.Println("----------------------------")
			for _, input := range neuron.connections {
				fmt.Println("Connection: ", input.origin.name, " - ", input.weight)
			}
		}
	}

}

func (n *Net) randomizeWeights() {

	for _, layer := range n.layers {
		for _, neuron := range layer.neurons {
			for _, input := range neuron.connections {
				input.weight = rand.Float64()

			}
		}
	}

}

func (n *Net) processInput(inputValues []float64) []float64 {

	//procesa la primera capa de neuronas, poniéndole solo los valores como outputs
	inputLayer := n.layers[0]
	for i, neuron := range inputLayer.neurons {
		neuron.output = inputValues[i]
	}

	//procesa el resto de las capas propagando los valores
	for _, l := range n.layers {

		for _, neuron := range l.neurons {
			neuron.calculateOutput()
		}

	}

	outputLayer := n.layers[len(n.layers)-1]

	outputValues := []float64{}

	for _, neu := range outputLayer.neurons {
		outputValues = append(outputValues, neu.output)
	}

	return outputValues

}

////////////////////////////////////////////////////////////////
type Layer struct {
	neurons []*Neuron
	name    string
}

func (l *Layer) addNeurons(quantity int) {
	for i := 0; i < quantity; i++ {
		neuron := &Neuron{name: (l.name + "-" + strconv.Itoa(i))}
		l.neurons = append(l.neurons, neuron)
	}
}

func (this *Layer) connectToLayer(l *Layer) {

	//agarra cada una de las neuronas de la capa actual
	for _, originNeuron := range this.neurons {

		//para cada una de las neuronas, crea una conexión con cada neurona de la otra capa
		for _, destinyNeuron := range l.neurons {

			//crea una conexión
			w := &Connection{origin: originNeuron, destiny: destinyNeuron, weight: 0}
			destinyNeuron.connections = append(destinyNeuron.connections, w)

		}

	}

}

////////////////////////////////////////////////////////////////
type Connection struct {
	origin  *Neuron
	destiny *Neuron
	weight  float64
}

//Help
//https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
//https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
//http://datathings.com/blog/post/neuralnet/
//https://becominghuman.ai/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
//https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
