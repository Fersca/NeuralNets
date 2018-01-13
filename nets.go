package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
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
		net := createNet(2,2,1)
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

	originanlCanti := len(generation)
	//ordena de menor a mayor
	ordena(generation)
	fmt.Println("best: ", generation[0].diff)
	fmt.Println("worst: ", generation[len(generation)-1].diff)

	for i:=0; i<100; i++{
		//borra 2
		generation = deleteWorstElements(generation, len(generation)/5)

		//procrea
		children := createChildren(generation)

		//calcula el test sobre la nueva generación
		calculateDiffinGeneration(children, testCases)

		ordena(children)
		
		//mezcla las generaciones
		generation = mergeElements(generation, children)

		//ordena de menor a mayor
		ordena(generation)

		//borra peores
		generation = deleteWorstElements(generation, len(generation)-originanlCanti)

		//printGeneration(generation)
		//fmt.Println(averageGeneration(generation))
		fmt.Println("best: ", generation[0].diff)
	
	}

}

func printGeneration(generation []*Net){
	for _,n := range generation {
		fmt.Println(n.diff)
	}
	
}

func createNet(input int, hidden int, output int) *Net {
	net := &Net{}
	net.input = input
	net.hidden = hidden
	net.output = output

	inputLayer := &Layer{name: "Input Layer"}
	inputLayer.addNeurons(input)

	hiddenLayer := &Layer{name: "Hidden Layer"}
	hiddenLayer.addNeurons(hidden)

	outputLayer := &Layer{name: "Output Layer"}
	outputLayer.addNeurons(output)

	inputLayer.connectToLayer(hiddenLayer)
	hiddenLayer.connectToLayer(outputLayer)

	net.addLayer(inputLayer)
	net.addLayer(hiddenLayer)
	net.addLayer(outputLayer)

	net.randomizeWeights()

	return net
}

func mergeElements(generation []*Net, children []*Net) []*Net {
	for _, n := range children {
		generation = append(generation, n)
	}
	return generation
}

func ordena(generation []*Net) {
	sort.Slice(generation, func(i, j int) bool {
		return generation[i].diff < generation[j].diff
	})	
}


func createChildren(generation []*Net) []*Net {

	children := []*Net{}

	tamanio := len(generation)/5

	for i:=0;i<tamanio;i++{
		for j:=i;j<tamanio;j++{
			n := generation[i]
			m := generation[j]
			//los procrea			
			son := createSon(n, m)
			//los guarda en la colección
			children = append(children, son)			
		}
	}

	return children

}

func printWeight(net1 *Net) {
	var suma float64
	for _, layer := range net1.layers {
		for _, neuron := range layer.neurons {
			for _, input := range neuron.connections {
				suma = suma + input.weight
			}
		}
	}
	fmt.Println("suma: ", suma)
}
func createSon(net1 *Net, net2 *Net) *Net {

	pesos1 := []float64{}
	pesos2 := []float64{}
	
	for _, layer := range net1.layers {
		for _, neuron := range layer.neurons {
			for _, input := range neuron.connections {
				pesos1 = append(pesos1, input.weight)
			}
		}
	}

	for _, layer := range net2.layers {
		for _, neuron := range layer.neurons {
			for _, input := range neuron.connections {
				pesos2 = append(pesos2, input.weight)
			}
		}
	}

	net3 := createNet(net1.input, net1.hidden, net1.output)

	i:=0
	for _, layer := range net3.layers {
		for _, neuron := range layer.neurons {
			for _, input := range neuron.connections {
				input.weight = mutaGen(pesos1[i], pesos2[i])			
				i++	
			}
		}
	}

	return net3
}

func mutaGen(gen1 float64, gen2 float64) float64 {

	if rand.Float64()<0.05 {
		return rand.Float64()
	} 
	if rand.Float64()<0.5 {
			return gen1
	}
	return gen2

}

func deleteWorstElements(generation []*Net, cant int) []*Net {

	cleanGeneration := []*Net{}

	for id, n := range generation {
		if id < (len(generation) - cant) {
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
			diff := math.Abs(output[0] - t.output[0])
			//fmt.Println("result: ", output[0], "target: ", t.output[0], "diff: ", diff)
			totalDiff = totalDiff + diff
		}

		//me guardo la direrencia total en la red
		n.diff = totalDiff / float64(len(testCases))

	}

}

func averageGeneration(generation []*Net) float64 {
	var suma float64
	for _,n := range generation {
		suma = suma + n.diff
	}
	return suma / float64(len(generation))

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
	input int
	hidden int
	output int
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
//https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f
