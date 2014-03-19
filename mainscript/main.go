package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/blas/goblas"

	"github.com/gonum/matrix/mat64"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
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
	var callerType string
	flag.StringVar(&callerType, "caller", "serial", "what kind of data caller to be used with SU2")

	flag.Parse()

	runtime.GOMAXPROCS(nCpu)
	mat64.Register(goblas.Blas{})
	dbw.Register(golas.Blas{})

	var caller driver.SU2Syscaller
	// Get the SU2Caller
	switch callerType {
	default:
		log.Fatal("caller " + callerType + " not recognized")
	case "serial":
		caller = driver.Serial{true}
	case "cluster":
		driver.Cluster{4, true}
	}

	set, err := settings.GetSettings(
		training,
		testing,
		features,
		weight,
		algorithm,
		convergence,
		caller,
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
