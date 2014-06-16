package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"runtime"
	"time"

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

	rand.Seed(time.Now().UnixNano())
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

	caller := driver.Serial{}

	//nRuns := len(testTrainPairs) * len(algorithms) * len(weights) * len(features) * len(convergence)

	// Construct all of the datasets

	var sets []*ransuq.Settings
	settingCases := GetCases()
	fmt.Println("The number of runs that will be done is: ", len(settingCases))
	for _, c := range settingCases {

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
			log.Fatal(err)
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
		sets = append(sets, set)
	}

	scheduler := ransuq.NewLocalScheduler()
	errs := ransuq.MultiTurb(sets, scheduler)

	for _, err := range errs {
		if err != nil {
			fmt.Println("Finished with error")
			fmt.Println(errs)
		}
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

func GetCases() []*settingCase {
	return []*settingCase{
		/*
			{
				Name:         "Nondim Production for single flatplate",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.SingleRae,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},

			{
				Name:         "Nondim Production for multi flatplate",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.SingleRae,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},

			{
				Name:         "Nondim Production for multi flatplate BL",
				TrainingData: settings.MultiFlatplateBL,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.FlatplateBlOnlyCutoff},
			},

			{
				Name:         "Nondim Destruction for single flatplate",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimDestruction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
			{
				Name:         "Nondim Destruction for multi flatplate",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimDestruction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
			{
				Name:         "Nondim Destruction for multi flatplate BL",
				TrainingData: settings.MultiFlatplateBL,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimDestruction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.FlatplateBlOnlyCutoff},
			},

			{
				Name:         "Nondim CrossProduction for single flatplate",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimCrossProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},

			{
				Name:         "Nondim CrossProduction for multi flatplate",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimCrossProduction,
				Convergence:  settings.TenKIter,

				ExtraString: []string{settings.NoExtraStrings},
			},

			{
				Name:         "Nondim CrossProduction for multi flatplate BL",
				TrainingData: settings.MultiFlatplateBL,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimCrossProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.FlatplateBlOnlyCutoff},
			},
		*/

		/*
			{
				Name:         "Nondim Source for single flatplate",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimSource,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/
		/*
			{
				Name:         "Nondim CrossProduction for multi flatplate",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimSource,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
			{
				Name:         "Nondim CrossProduction for multi flatplate BL",
				TrainingData: settings.MultiFlatplateBL,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimSource,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.FlatplateBlOnlyCutoff},
			},
		*/

		/*
			{
				Name:         "Learn dimensional cross production",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.NoDataset,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.CrossProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{},
			},
		*/

		/*
			{
				Name:         "Test Learn scaled production",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.SingleFlatplate,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Production,
				Convergence:  settings.OneHundIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/
		/*
			{
				Name:         "Learn scaled production",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Production,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/
		/*
			{
				Name:         "Learn scaled destruction",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Destruction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/

		/*
			{
				Name:         "Learn scaled cross production",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.CrossProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/
		/*
			{
				Name:         "Test Learn scaled full source",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.FlatplateSweep,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Source,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/

		/*
			{
				Name:         "Learn scaled full source",
				TrainingData: settings.MultiFlatplate,
				TestingData:  settings.NoDataset,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Source,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/

		/*
			{
				Name:         "Nondim CrossProduction for single flatplate",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.MultiFlatplate,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimCrossProduction,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/

		/*
			{
				Name:         "Nondim Source for single flatplate",
				TrainingData: settings.SingleFlatplate,
				TestingData:  settings.SingleFlatplate,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.NondimSource,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.NoExtraStrings},
			},
		*/
		/*
			{
				Name:         "LES tenth of Data for Fw only in BL",
				TrainingData: settings.LES4Tenth,
				TestingData:  settings.SingleFlatplate,
				Algorithm:    settings.NetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.LES4Tenth,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{settings.FlatplateBlOnlyCutoff},
			},
		*/
		/*
			{
				Name:         "Single NACA 0012 test case",
				TrainingData: settings.MultiNaca0012,
				TestingData:  settings.Naca0012Sweep,
				Algorithm:    settings.MulNetTwoFifty,
				Weight:       settings.NoWeight,
				Features:     settings.Source,
				Convergence:  settings.TenKIter,
				ExtraString:  []string{},
			},
		*/
		{
			Name:         "Pressure driven wall",
			TrainingData: settings.PressureGradientMulti,
			TestingData:  settings.NoDataset,
			Algorithm:    settings.MulNetTwoFifty,
			Weight:       settings.NoWeight,
			Features:     settings.Source,
			Convergence:  settings.TenKIter,
			ExtraString:  []string{},
		},
	}
}
