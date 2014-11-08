package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"

	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/mlalg"
	"github.com/btracey/ransuq/settings"

	"github.com/gonum/blas/dbw"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"

	"github.com/btracey/su2tools/driver"

	"github.com/davecheney/profile"
)

func init() {
	mat64.Register(goblas.Blas{})
	dbw.Register(goblas.Blas{})
	//mat64.Register(cblas.Blas{})
	//dbw.Register(cblas.Blas{})

	//	rand.Seed(time.Now().UnixNano())
}

func main() {
	var location string
	flag.StringVar(&location, "location", "local", "where is the code being run (local, cluster)")
	var doprofile bool
	flag.BoolVar(&doprofile, "profile", false, "should the code be profiled")
	var casefile string
	flag.StringVar(&casefile, "j", "none", "json file for which case to run")
	flag.Parse()

	if casefile == "none" {
		log.Fatal("No case json file specified")
	}

	switch location {
	case "local":
		//runtime.GOMAXPROCS(1)
		runtime.GOMAXPROCS(runtime.NumCPU() - 2) // leave some CPU open so the computer doesn't crash
	case "cluster":
		fmt.Println("Cluster num CPU is", runtime.NumCPU())
		runtime.GOMAXPROCS(runtime.NumCPU())
	default:
		log.Fatal("unknown location")
	}

	if doprofile {
		defer profile.Start(profile.CPUProfile).Stop()
	}

	caller := driver.Serial{} // Run the SU^2 cases in serial

	// Construct all of the datasets

	var sets []*ransuq.Settings

	f, err := os.Open(casefile)
	if err != nil {
		log.Fatal("error opening case file:", err)
	}

	settingCases := GetCases(f)
	fmt.Println("The number of runs that will be done is: ", len(settingCases))
	for i, c := range settingCases {

		fmt.Println("Set testing data", c.TestingData)
		set, err := settings.GetSettings(
			c.TrainingData,
			c.TestingData,
			c.Features,
			c.Weight,
			c.Algorithm,
			c.Convergence,
			caller,
			c.ExtraString, // Need to pass all of them because don't want to double do training
		)
		if err != nil {
			log.Fatal("error getting settings:", err)
		}
		if c.Algorithm == settings.MulNetTwoFifty {
			os := &mlalg.MulOutputScaler{}
			is := &mlalg.MulInputScaler{
				Scaler:          set.Trainer.InputScaler,
				MulOutputScaler: os,
			}
			set.Trainer.InputScaler = is
			set.Trainer.OutputScaler = os
		}
		if len(set.TrainingData) == 0 {
			log.Fatal("no training data in set ", i)
		}
		sets = append(sets, set)
	}

	scheduler := ransuq.NewLocalScheduler()
	fmt.Println("Begin ransuq.MultiTurb")
	errs := ransuq.MultiTurb(sets, scheduler)
	fmt.Println("End ransuq.MultiTurb")

	var haserror bool
	for _, err := range errs {
		if err != nil {
			haserror = true
			fmt.Println("Finished with error")
			fmt.Println(errs)
		}
	}
	if haserror {
		return
	}

	fmt.Println("Finished without error")

}

type settingCase struct {
	Name         string
	TrainingData string
	TestingData  string
	Algorithm    string
	Weight       string
	Features     string
	Convergence  string
	ExtraString  []string
}

func GetCases(r io.Reader) []*settingCase {
	c := make([]*settingCase, 0, 100)

	dec := json.NewDecoder(r)
	err := dec.Decode(&c)
	if err != nil {
		log.Fatal("error decoding cases:", err)
	}
	fmt.Println("Done decoding")
	for i := range c {
		fmt.Println(c[i])
	}
	return c
}
