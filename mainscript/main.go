package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	"github.com/gonum/blas/cblas"
	"github.com/gonum/blas/dbw"

	"github.com/gonum/matrix/mat64"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
)

func main() {

	var training string
	flag.StringVar(&training, "train", "", "What training data sets to use")
	var testing string
	flag.StringVar(&testing, "test", "none", "What testing data sets to use")
	var features string
	flag.StringVar(&features, "features", "", "What input/output feature sets to use")
	var weight string
	flag.StringVar(&weight, "weights", string(settings.NoWeight), "What weight function should be used")
	var nCpu int
	flag.IntVar(&nCpu, "nCpu", runtime.NumCPU(), "How many CPU should the program use")
	var algorithm string
	flag.StringVar(&algorithm, "alg", string(settings.NetTwoFifty), "what learning algorithm should be used")
	var convergence string
	flag.StringVar(&convergence, "convergence", string(settings.StandardTraining), "how long should the machine learning model be run")

	flag.Parse()

	runtime.GOMAXPROCS(nCpu)
	mat64.Register(cblas.Blas{})
	dbw.Register(cblas.Blas{})

	set, err := settings.GetSettings(
		training,
		testing,
		features,
		weight,
		algorithm,
		convergence,
	)

	if err != nil {
		log.Fatal(err)
	}

	// Run the machine learning

	err = ransuq.MlTurb(set)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Finished running")
}
