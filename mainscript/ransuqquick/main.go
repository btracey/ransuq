package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"

	"github.com/gonum/blas/cblas"
	"github.com/gonum/blas/dbw"
	"github.com/gonum/matrix/mat64"

	"github.com/btracey/su2tools/driver"

	"github.com/davecheney/profile"
)

func init() {
	//mat64.Register(goblas.Blas{})
	//dbw.Register(goblas.Blas{})
	mat64.Register(cblas.Blas{})
	dbw.Register(cblas.Blas{})
}

func main() {

	var location string
	flag.StringVar(&location, "location", "local", "where is the code being run (local, cluster)")
	var doprofile bool
	flag.BoolVar(&doprofile, "profile", false, "should the code be profiled")
	flag.Parse()

	switch location {
	case "local":
		runtime.GOMAXPROCS(runtime.NumCPU() - 2) // leave some CPU open so the computer doesn't crash
	case "cluster":
		runtime.GOMAXPROCS(runtime.NumCPU())
	default:
		log.Fatal("unknown location")
	}

	if doprofile {
		defer profile.Start(profile.CPUProfile).Stop()
	}

	testTrainPairs := [][2]string{
		//{settings.MultiFlatplate, settings.FlatplateSweep},
		//{settings.MultiFlatplateBL, settings.FlatplateSweep},
		//{settings.SingleRae, settings.SingleRae},
		//{settings.SyntheticFlatplateProduction, settings.FlatplateSweep},
		//{settings.MultiAndSynthFlatplate, settings.FlatplateSweep},
		//{settings.MultiAndSynthFlatplate, settings.NoDataset},
		//{settings.MultiAndSynthFlatplate, settings.SingleFlatplate},
		{settings.LES4, settings.NoDataset},
	}
	algorithms := []string{
		"net_2_50",
		//"net_1_50",
	}
	weights := []string{"none"}
	features := []string{
		//"nondim_production",
		//"nondim_production_log",
		//"nondim_production_logchi",
		//"nondim_destruction",
		//"nondim_crossproduction",
		//"nondim_source",
		//"production",
		//"destruction",
		settings.FwLES,
	}
	convergence := []string{
		//settings.StandardTraining,
		settings.FiveKIter,
	}
	//convergence := []string{settings.QuickTraining}

	caller := driver.Serial{}

	nRuns := len(testTrainPairs) * len(algorithms) * len(weights) * len(features) * len(convergence)
	fmt.Println("The number of runs that will be done is: ", nRuns)

	// Construct all of the datasets

	var sets []*ransuq.Settings

	for _, pair := range testTrainPairs {
		for _, feature := range features {
			for _, weight := range weights {
				for _, alg := range algorithms {
					for _, conv := range convergence {
						set, err := settings.GetSettings(
							pair[0],
							pair[1],
							feature,
							weight,
							alg,
							conv,
							caller,
						)
						if err != nil {
							panic(err)
						}
						sets = append(sets, set)
					}
				}
			}
		}
	}
	scheduler := ransuq.NewLocalScheduler()
	err := ransuq.MultiTurb(sets, scheduler)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println("Finished without error")

}
