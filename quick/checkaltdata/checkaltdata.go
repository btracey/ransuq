package main

import (
	"fmt"
	"log"
	"math"

	"github.com/btracey/ransuq/quickload"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
)

func init() {
	mat64.Register(goblas.Blas{})
}

func main() {
	dataset := "oneram6"
	features := "fw_dim_alt"
	fmt.Println("starting load")
	inputData, outputData, weights, err := quickload.Load(dataset, features)
	if weights != nil {
		log.Fatal(err)
	}
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("done load")
	inputFeatures, outputFeatures, err := settings.GetFeatures(features)
	if err != nil {
		log.Fatal(err)
	}

	Chi := findStringLocation(inputFeatures, "Chi")
	OmegaAlt := findStringLocation(inputFeatures, "OmegaAlt")
	Dist := findStringLocation(inputFeatures, "WallDistance")

	Fw := findStringLocation(outputFeatures, "Fw")

	chis := inputData.Col(nil, Chi)
	omegaAlts := inputData.Col(nil, OmegaAlt)
	dists := inputData.Col(nil, Dist)

	fws := outputData.Col(nil, Fw)

	for i := range chis {
		chi := chis[i]
		omegaAlt := omegaAlts[i]
		dist := dists[i]

		SA := sa.SA{}
		SA.Nu = 1
		SA.NuHat = chi
		SA.Chi = chi
		SA.Omega = omegaAlt
		SA.WallDistance = dist

		SA.ComputeFw()
		if math.Abs(SA.Fw-fws[i]) > 1e-13 {
			fmt.Println(SA.Fw, fws[i])
			log.Fatal("bad point")
		}
	}
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}
