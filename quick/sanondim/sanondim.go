package main

import (
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"github.com/btracey/ransuq"
	"github.com/btracey/ransuq/settings"
	"github.com/btracey/su2tools/driver"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

var (
	gopath   string
	basepath string
)

func init() {
	mat64.Register(goblas.Blas{})

	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		log.Fatal("no gopath")
	}
	basepath = filepath.Join(gopath, "results", "ransuq", "sanondim")
}

var inputFeatures = []string{
	"SourceNondimer", "Source", "Fw", "NuHat", "OmegaBar", "OmegaNondimer", "Nu",
	"DUDX", "DVDX", "DUDY", "DVDY", "DNuHatDX", "DNuHatDY", "WallDistance", "NondimSource",
	"Chi",
}

var (
	SourceNondimer = findStringLocation(inputFeatures, "SourceNondimer")
	Source         = findStringLocation(inputFeatures, "Source")
	NondimSource   = findStringLocation(inputFeatures, "NondimSource")
	Fw             = findStringLocation(inputFeatures, "Fw")
	NuHat          = findStringLocation(inputFeatures, "NuHat")
	OmegaBar       = findStringLocation(inputFeatures, "OmegaBar")
	OmegaNondimer  = findStringLocation(inputFeatures, "OmegaNondimer")
	Nu             = findStringLocation(inputFeatures, "Nu")
	DNuHatDX       = findStringLocation(inputFeatures, "DNuHatDX")
	DNuHatDY       = findStringLocation(inputFeatures, "DNuHatDY")
	DUDX           = findStringLocation(inputFeatures, "DUDX")
	DUDY           = findStringLocation(inputFeatures, "DUDY")
	DVDX           = findStringLocation(inputFeatures, "DVDX")
	DVDY           = findStringLocation(inputFeatures, "DVDY")
	Chi            = findStringLocation(inputFeatures, "Chi")
	WallDistance   = findStringLocation(inputFeatures, "WallDistance")
)

var (
	nuair = 1.48e-5
	//nuair = 1
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU() - 2)
	// Set this up for a loop of datasets. Need to fix plotting places.

	datasetStrs := []string{
		"single_flatplate", "single_naca_0012", "single_naca_0012_bl", "single_flatplate_bl",
		"oneram6",
	}
	//var wg sync.WaitGroup
	for _, setname := range datasetStrs {
		//wg.Add(1)
		//go func(setname string) {
		//wg.Done()
		// Load data

		datasets, err := settings.GetDatasets(setname, driver.Serial{})
		if err != nil {
			log.Fatal(err)
		}

		inputDataMat, _, _, err := ransuq.DenseLoadAll(datasets, inputFeatures, nil, nil, nil)
		if err != nil {
			log.Fatal(err)
		}

		inputData := inputDataMat.(*mat64.Dense)
		nSamples, _ := inputData.Dims()

		datasetPath := filepath.Join(basepath, setname)
		err = os.MkdirAll(datasetPath, 0700)
		if err != nil {
			log.Fatal(err)
		}
		// Plot histograms of current data
		idxs := []int{SourceNondimer, Fw, Nu, NuHat, Source, DUDY, WallDistance,
			DNuHatDX, DNuHatDY, OmegaBar, NondimSource, Chi}
		for _, idx := range idxs {
			pltName := filepath.Join(datasetPath, inputFeatures[idx])
			pltName += ".jpg"
			data := inputData.Col(nil, idx)
			makeHistogram(data, pltName)
		}

		nu := inputData.Col(nil, Nu)
		nuHat := inputData.Col(nil, NuHat)
		nuTilde := make([]float64, nSamples)
		copy(nuTilde, nuHat)
		for i := range nuTilde {
			nuTilde[i] = nu[i] / nuair
		}
		makeHistogram(nuTilde, filepath.Join(datasetPath, "NuTilde.jpg"))

		nuHatTilde := make([]float64, nSamples)
		floats.DivTo(nuHatTilde, nuHat, nuTilde)
		makeHistogram(nuHatTilde, filepath.Join(datasetPath, "NuHatTilde.jpg"))

		nuHatTildeStar := make([]float64, nSamples)
		for i := range nuHatTildeStar {
			nuHatTildeStar[i] = nuHatTilde[i] + 3*nuair
		}
		makeHistogram(nuHatTildeStar, filepath.Join(datasetPath, "NuHatTildeStar.jpg"))
		dist := inputData.Col(nil, WallDistance)
		distStar := make([]float64, nSamples)
		for i := range distStar {
			distStar[i] = math.Min(dist[i], 300*nuair)
			//distStar[i] = math.Min(, y)
		}
		makeHistogram(distStar, filepath.Join(datasetPath, "DistStar.jpg"))

		omegaBar := inputData.Col(nil, OmegaBar)
		omegaNondimer := inputData.Col(nil, OmegaNondimer)
		omega := make([]float64, nSamples)
		floats.MulTo(omega, omegaBar, omegaNondimer)
		omegaTilde := make([]float64, nSamples)
		floats.DivTo(omegaTilde, omega, nuTilde)

		makeHistogram(omegaTilde, filepath.Join(datasetPath, "OmegaTilde.jpg"))

		omegaNondimerTilde := make([]float64, nSamples)
		for i := range omegaNondimerTilde {
			omegaNondimerTilde[i] = nuHatTildeStar[i] / (distStar[i] * distStar[i])
		}
		makeHistogram(omegaNondimerTilde, filepath.Join(datasetPath, "OmegaNondimerTilde.jpg"))

		invOmegaNondimerTilde := make([]float64, nSamples)
		for i := range invOmegaNondimerTilde {
			invOmegaNondimerTilde[i] = 1 / omegaNondimerTilde[i]
		}
		makeHistogram(invOmegaNondimerTilde, filepath.Join(datasetPath, "InvOmegaNondimerTilde.jpg"))

		nondimOmegaTilde := make([]float64, nSamples)
		floats.DivTo(nondimOmegaTilde, omegaTilde, omegaNondimerTilde)

		makeHistogram(nondimOmegaTilde, filepath.Join(datasetPath, "NondimOmegaTilde.jpg"))

		source := inputData.Col(nil, Source)
		sourceTilde := make([]float64, nSamples)
		for i := range sourceTilde {
			sourceTilde[i] = source[i] / (nuTilde[i] * nuTilde[i])
		}
		makeHistogram(sourceTilde, filepath.Join(datasetPath, "SourceTilde.jpg"))
		sourceNondimerTilde := make([]float64, nSamples)
		for i := range sourceNondimerTilde {
			sourceNondimerTilde[i] = (nuHatTildeStar[i] / distStar[i]) * (nuHatTildeStar[i] / distStar[i])
		}
		makeHistogram(sourceTilde, filepath.Join(datasetPath, "SourceNondimerTilde.jpg"))
		nondimSourceTilde := make([]float64, nSamples)
		floats.DivTo(nondimSourceTilde, sourceTilde, sourceNondimerTilde)
		makeHistogram(nondimSourceTilde, filepath.Join(datasetPath, "NondimSourceTilde.jpg"))

		invSourceNondimerTilde := make([]float64, nSamples)
		for i := range sourceNondimerTilde {
			invSourceNondimerTilde[i] = 1 / sourceNondimerTilde[i]
		}
		makeHistogram(invSourceNondimerTilde, filepath.Join(datasetPath, "invSourceNondimerTilde.jpg"))

		/*


			tmp := make([]float64, nSamples)
			// Plot source over nu
			source := inputData.Col(nil, Source)
			nu := inputData.Col(nil, Nu)
			floats.MulTo(tmp, source, nu)
			makeHistogram(tmp, filepath.Join(datasetPath, "sourcenu.jpg"))

			nuhat := inputData.Col(nil, NuHat)
			invnuhat2 := inputData.Col(nil, NuHat)
			for i, v := range invnuhat2 {
				invnuhat2[i] = (nuair / v) * (nuair / v)
			}
			min, _ := floats.Min(nuhat)
			fmt.Println("minnuhat = ", min)
			floats.MulTo(tmp, source, invnuhat2)
			makeHistogram(tmp, filepath.Join(datasetPath, "sourcenuhat2.jpg"))

			nuhatalt := inputData.Col(nil, NuHat)
			for i := range nuhatalt {
				nuhatalt[i] += 3 * nu[i]
			}
			min, _ = floats.Min(nuhatalt)
			fmt.Println("minnuhatalt ", min)

			omega := make([]float64, nSamples)
			omeganondimer := inputData.Col(nil, OmegaNondimer)
			omegabar := inputData.Col(nil, OmegaBar)
			floats.MulTo(omega, omeganondimer, omegabar)
			makeHistogram(omega, filepath.Join(datasetPath, "omega.jpg"))

			sourceEst := make([]float64, nSamples)
			// Compute spalart allmaras source
			for i := 0; i < nSamples; i++ {
				SA := sa.SA{
					NDim:     2,
					Nu:       inputData.At(i, Nu),
					NuHat:    inputData.At(i, NuHat),
					DNuHatDX: []float64{inputData.At(i, DNuHatDX), inputData.At(i, DNuHatDY)},
					DUIdXJ: [][]float64{{inputData.At(i, DUDX), inputData.At(i, DVDX)},
						{inputData.At(i, DUDY), inputData.At(i, DVDY)}},
					WallDistance: inputData.At(i, WallDistance),
				}
				SA.Source()
				sourceEst[i] = SA.SourceTerm
			}
			for i := 0; i < nSamples; i++ {
				if math.Abs(sourceEst[i]-source[i]) > 1e-13 {
					log.Fatal("mismatch")
				}
			}

			invNondimer := make([]float64, nSamples)
			for i := range invNondimer {
				invNondimer[i] = 1.0 / inputData.At(i, SourceNondimer)
			}
			makeHistogram(invNondimer, filepath.Join(datasetPath, "invNondimer.jpg"))

			// What if we nondimensionalize by OmegaNuHat
			nondimAlt := make([]float64, nSamples)
			floats.MulTo(nondimAlt, omega, nuhat)
			for i := range nondimAlt {
				nondimAlt[i] = math.Abs(nondimAlt[i])
			}
			floats.AddConst(0, nondimAlt)
			makeHistogram(nondimAlt, filepath.Join(datasetPath, "SourceNondimerAlt.jpg"))

			nondimSourceAlt := make([]float64, nSamples)
			floats.DivTo(nondimSourceAlt, source, nondimAlt)
			makeHistogram(nondimSourceAlt, filepath.Join(datasetPath, "nondimSourceAlt.jpg"))
			//	}(setname)

			// Cap distance
			maxDist := 100 * 3 * nu[0] // Chi = 100 is threshhold
			altWallDist := inputData.Col(nil, WallDistance)
			for i, v := range altWallDist {
				altWallDist[i] = math.Min(v, maxDist)
			}

			nondimAlt2 := make([]float64, nSamples)
			floats.DivTo(nondimAlt2, nuhat, altWallDist)
			for i, v := range nondimAlt2 {
				nondimAlt2[i] *= v
			}
			invNondimAlt2 := make([]float64, nSamples)
			for i, v := range nondimAlt2 {
				invNondimAlt2[i] = 1 / v
			}
			makeHistogram(invNondimAlt2, filepath.Join(datasetPath, "invNondimerAlt2.jpg"))

			nondimAlt3 := make([]float64, nSamples)
			floats.DivTo(nondimAlt3, nuhatalt, altWallDist)
			for i, v := range nondimAlt3 {
				nondimAlt3[i] *= v
			}
			makeHistogram(nondimAlt, filepath.Join(datasetPath, "SourceNondimerAlt3.jpg"))

			invNondimAlt3 := make([]float64, nSamples)
			for i, v := range nondimAlt3 {
				invNondimAlt3[i] = 1 / v
			}
			makeHistogram(invNondimAlt3, filepath.Join(datasetPath, "invNondimerAlt3.jpg"))

			nondimSourceAlt3 := make([]float64, nSamples)
			floats.DivTo(nondimSourceAlt3, source, nondimAlt3)
			makeHistogram(nondimSourceAlt3, filepath.Join(datasetPath, "nondimSourceAlt3.jpg"))

			// Need nondim Omega3
			omegaNondimerAlt3 := make([]float64, nSamples)
			for i := range omegaNondimerAlt3 {
				omegaNondimerAlt3[i] = nuhatalt[i] / (altWallDist[i] * altWallDist[i])
			}
			makeHistogram(omegaNondimerAlt3, filepath.Join(datasetPath, "omegaNondimerAlt3.jpg"))

			omegaBarAlt3 := make([]float64, nSamples)
			floats.DivTo(omegaBarAlt3, omega, omegaNondimerAlt3)
			makeHistogram(omegaBarAlt3, filepath.Join(datasetPath, "OmegaBarAlt3.jpg"))

		*/
	}
	//wg.Wait()
}

func makeHistogram(data []float64, name string) {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	h, err := plotter.NewHist(plotter.Values(data), 1000)
	if err != nil {
		log.Fatal(err)
	}
	p.Add(h)
	err = p.Save(4, 4, name)
	if err != nil {
		log.Fatal(err)
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
