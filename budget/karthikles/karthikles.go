package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"code.google.com/p/plotinum/plot"

	"github.com/btracey/numcsv"
	"github.com/btracey/quickplot"
	"github.com/btracey/ransuq/internal/util"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/blas/goblas"
	"github.com/gonum/matrix/mat64"
	"github.com/gonum/stat"
)

var gopath string

func init() {
	gopath = os.Getenv("GOPATH")
	if gopath == "" {
		log.Fatal("gopath not set")
	}
}

func init() {
	mat64.Register(goblas.Blas{})
}

var (
	newHeadings = []string{"Nu", "NuHat", "Chi", "Omega", "WallDistance", "NuGradMag",
		"Source", "NuHatAlt", "OmegaAlt", "NuGradMagAlt", "SourceAlt", "Fw"}

	NuNew              = util.FindStringLocation(newHeadings, "Nu")
	NuHatNew           = util.FindStringLocation(newHeadings, "NuHat")
	NuHatNewAlt        = util.FindStringLocation(newHeadings, "NuHatAlt")
	ChiNew             = util.FindStringLocation(newHeadings, "Chi")
	OmegaNew           = util.FindStringLocation(newHeadings, "Omega")
	OmegaNewAlt        = util.FindStringLocation(newHeadings, "OmegaAlt")
	WallDistanceNew    = util.FindStringLocation(newHeadings, "WallDistance")
	NuHatGradMagNew    = util.FindStringLocation(newHeadings, "NuGradMag")
	NuHatGradMagNewAlt = util.FindStringLocation(newHeadings, "NuGradMagAlt")
	SourceNew          = util.FindStringLocation(newHeadings, "Source")
	SourceNewAlt       = util.FindStringLocation(newHeadings, "SourceAlt")
	FwNew              = util.FindStringLocation(newHeadings, "Fw")
	//SourceDiffNew      = util.FindStringLocation(newHeadings, "SourceDiff")
	//SourceDiffAltNew   = util.FindStringLocation(newHeadings, "SourceDiffAlt")
)

func main() {
	nX := 1794
	nY := 161
	ignore := 3
	//rho := 1
	nu := 1e-3

	filename := filepath.Join(gopath, "data", "ransuq", "les_karthik", "sadata.txt")
	newfilename := filepath.Join(gopath, "data", "ransuq", "les_karthik", "sadatacomputed.txt")
	f, err := os.Open(filename)
	defer f.Close()
	if err != nil {
		log.Fatal(err)
	}

	read := numcsv.NewReader(f)
	headings, err := read.ReadHeading()
	if err != nil {
		log.Fatal(err)
	}
	data, err := read.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(headings)
	rows, cols := data.Dims()
	fmt.Println(rows, cols)

	XIdx := util.FindStringLocation(headings, "XIdx")
	YIdx := util.FindStringLocation(headings, "YIdx")
	DUDX := util.FindStringLocation(headings, "DUDX")
	DUDY := util.FindStringLocation(headings, "DUDY")
	DVDX := util.FindStringLocation(headings, "DVDX")
	DVDY := util.FindStringLocation(headings, "DVDY")
	DNuHatDX := util.FindStringLocation(headings, "DMutDX")
	DNuHatDY := util.FindStringLocation(headings, "DMutDY")
	NuT := util.FindStringLocation(headings, "MuT")
	XLoc := util.FindStringLocation(headings, "XLoc")
	YLoc := util.FindStringLocation(headings, "YLoc")
	UVel := util.FindStringLocation(headings, "UVel")
	VVel := util.FindStringLocation(headings, "VVel")
	//DNuHatDXX := util.FindStringLocation(headings, "dMuthatDXX")
	DNuHatDYY := util.FindStringLocation(headings, "DMutDYY")
	BLDelta := util.FindStringLocation(headings, "BLDelta")

	computedSource := make([]float64, rows)
	saSource := make([]float64, rows)
	saFw := make([]float64, rows)
	computedFw := make([]float64, rows)
	omega := make([]float64, rows)
	saSourceSub := make([]float64, 0)
	saFwSub := make([]float64, 0)

	newdata := mat64.NewDense(rows, len(headings), nil)

	count := 0
	for i := 0; i < rows; i++ {
		xidx := int(data.At(i, XIdx))
		yidx := int(data.At(i, YIdx))
		x := data.At(i, XLoc)
		y := data.At(i, YLoc)
		delta := data.At(i, BLDelta)
		u := data.At(i, UVel)
		v := data.At(i, VVel)
		dudx := data.At(i, DUDX)
		dudy := data.At(i, DUDY)
		dvdx := data.At(i, DVDX)
		dvdy := data.At(i, DVDY)
		nuhat := data.At(i, NuT)
		dNuHatDX := data.At(i, DNuHatDX)
		dNuHatDY := data.At(i, DNuHatDY)
		//dNuHatDXX := data.At(i, DNuHatDXX)
		dNuHatDYY := data.At(i, DNuHatDYY)
		dist := data.At(i, YLoc)

		SA := &sa.SA{
			NDim:         2,
			Nu:           nu,
			NuHat:        nuhat,
			DNuHatDX:     []float64{dNuHatDX, dNuHatDY},
			DUIdXJ:       [][]float64{{dudx, dvdx}, {dudy, dvdy}},
			WallDistance: dist,
		}
		saSource[i] = SA.Source()
		saFw[i] = SA.Fw
		omega[i] = SA.Omega

		nuHatGradMag := dNuHatDX*dNuHatDX + dNuHatDY*dNuHatDY

		conv := u*dNuHatDX + v*dNuHatDY
		diff := (1.0 / sa.Sigma) * ((nu+nuhat)*dNuHatDYY + dNuHatDX*dNuHatDY + dNuHatDY*dNuHatDY)
		computedSource[i] = conv - diff
		destDns := SA.Production + diff - conv
		//fact := (nuhat / (y + 1e-12)) * (nuhat / (y + 1e-12))
		factinv := (y / (nuhat + 1e-12)) * (y / (nuhat + 1e-12))
		fw := (destDns*factinv + sa.Cb1OverKappaSquared*SA.Ft2) / sa.CW1

		computedFw[i] = fw

		if y > 0.7*delta || y < 1e-6 || xidx <= ignore || xidx > nX-ignore || yidx <= ignore || yidx > nY-ignore {
			continue
		}
		if computedSource[i] < -0.09 {
			fmt.Println("low source ", i)
			fmt.Println("conv", conv)
			fmt.Println("diff", diff)
			fmt.Println("source =", computedSource[i])
			fmt.Println("x =", x)
			fmt.Println("y =", y)
			fmt.Printf("%#v\n", SA)
			os.Exit(1)
		}

		nuhatalt := sa.NuHatAlt(nu, nuhat)
		omegaalt := sa.OmegaAlt(omega[i], nu)
		nuHatGradMagAlt := sa.NuGradMagAlt(nuHatGradMag, nu)
		sourceAlt := sa.SourceAlt(computedSource[i], nu)
		newdata.Set(count, NuNew, nu)
		newdata.Set(count, NuHatNew, nuhat)
		newdata.Set(count, NuHatNewAlt, nuhatalt)
		newdata.Set(count, ChiNew, nuhat)
		newdata.Set(count, OmegaNew, omega[i])
		newdata.Set(count, OmegaNewAlt, omegaalt)
		newdata.Set(count, WallDistanceNew, y)
		newdata.Set(count, NuHatGradMagNew, nuHatGradMag)
		newdata.Set(count, NuHatGradMagNewAlt, nuHatGradMagAlt)
		newdata.Set(count, SourceNew, computedSource[i])
		newdata.Set(count, SourceNewAlt, sourceAlt)
		newdata.Set(count, FwNew, computedFw[i])
		saSourceSub = append(saSourceSub, saSource[i])
		saFwSub = append(saFwSub, saFw[i])
		count++
	}
	newdata.View(newdata, 0, 0, count, len(newHeadings))

	rowsNew, colsNew := newdata.Dims()
	fmt.Println("newrows", rowsNew, colsNew)

	fnew, err := os.Create(newfilename)
	if err != nil {
		log.Fatal(err)
	}
	w := numcsv.NewWriter(fnew)
	err = w.WriteAll(newHeadings, newdata)
	if err != nil {
		log.Fatal(err)
	}

	meanSA := stat.Mean(saSource, nil)
	stdSA := stat.StdDev(saSource, meanSA, nil)
	meanComp := stat.Mean(computedSource, nil)
	stdComp := stat.StdDev(computedSource, meanComp, nil)

	corr := stat.Correlation(computedSource, meanComp, stdComp, saSource, meanSA, stdSA, nil)
	fmt.Println("correlation: ", corr)

	if err = makescatter(computedSource, saSource, "sourcecomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makescatter(computedFw, saFw, "fwcomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makescatter(newdata.Col(nil, FwNew), saFwSub, "fwsubcomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makescatter(newdata.Col(nil, SourceNew), saSourceSub, "sourcesubcomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makescatter(newdata.Col(nil, OmegaNew), newdata.Col(nil, SourceNew), "omegacomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makescatter(newdata.Col(nil, ChiNew), newdata.Col(nil, SourceNew), "chicomp.jpg"); err != nil {
		log.Fatal(err)
	}
	if err = makecontour(newdata.Col(nil, ChiNew), newdata.Col(nil, OmegaNew), newdata.Col(nil, SourceNew), "sourcecontour.jpg", 0.05, 0.95); err != nil {
		log.Fatal(err)
	}
}

func makescatter(x, y []float64, name string) (err error) {
	if len(x) != len(y) {
		panic("slice length mismatch")
	}

	scatter, err := quickplot.Scatter(x, y)
	if err != nil {
		return err
	}
	p, err := plot.New()
	if err != nil {
		return err
	}
	p.Add(scatter)
	return p.Save(8, 8, name)
}

func makecontour(x, y, z []float64, name string, min, max float64) error {
	sortZ := make([]float64, len(z))
	copy(sortZ, z)
	sort.Float64s(sortZ)

	minVal := stat.Quantile(min, stat.Empirical, sortZ, nil)
	maxVal := stat.Quantile(max, stat.Empirical, sortZ, nil)

	fmt.Println(minVal)
	fmt.Println(maxVal)

	contour, err := quickplot.Contour(x, y, z)
	contour.SetMax(maxVal)
	contour.SetMin(minVal)
	p, err := plot.New()
	if err != nil {
		return err
	}
	p.Add(contour)
	return p.Save(8, 8, name)

}
