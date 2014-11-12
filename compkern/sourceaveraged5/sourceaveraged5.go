package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"
	"strconv"

	"github.com/btracey/ransuq"
	"github.com/reggo/reggo/supervised/nnet"
)

var (
	_ = nnet.Net{}
)

var (
	netname string
	outname string
)

func init() {
	flag.StringVar(&netname, "net", "trained_algorithm.json", "neural network json file")
	flag.StringVar(&outname, "out", "source.txt", "output filename for the source")
}

const (
	nArgs = 5
)

// This function computes the dimensionalized source as a function of
// Nu, NuHat, Omega, NuGradMag, and WallDistance
// Omega is computed as sqrt(2 * Wij * Wij)
func main() {
	flag.Parse()
	args := flag.Args()

	f, err := os.Create(outname)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if len(args) != nArgs {
		log.Print(args)
		log.Fatal("incorrect number of arguments")
	}

	nu := parse(args[0], "nu")
	nuhat := parse(args[1], "nuhat")
	omega := parse(args[2], "omega")
	nugradmag := parse(args[3], "nu grad mag")
	dist := parse(args[4], "wall distance")

	// The trained algorithm is a function of
	//  nuhatalt, omegaalt, nogradmagalt, walldistance
	input := []float64{nuhat / nu, omega / nu, nugradmag / (nu * nu), dist}
	output := make([]float64, 1)

	// Parse the neural network
	p := ransuq.ScalePredictor{}
	netfile, err := os.Open(netname)
	if err != nil {
		log.Fatalf("error opening net filename: %v", err)
	}
	defer netfile.Close()
	decoder := json.NewDecoder(netfile)
	err = decoder.Decode(&p)
	if err != nil {
		log.Fatalf("error loading net: %v", err)
	}

	output, err = p.Predict(input, output)
	if err != nil {
		log.Fatal("error predicting: %v", err)
	}
	if len(output) != 1 {
		panic("bad size")
	}
	source := output[0] * nu * nu

	vs := strconv.FormatFloat(source, 'g', 16, 64)
	f.WriteString(vs)
}

func parse(str, name string) float64 {
	v, err := strconv.ParseFloat(str, 64)
	if err != nil {
		log.Fatal("error parsing " + name + ": " + err.Error())
	}
	return v
}
