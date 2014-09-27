package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"

	"github.com/btracey/diff/scattered"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/floats"

	"github.com/btracey/su2tools/mesh"
)

// Data to load from SU2 Data
var features = []string{"Source", "Nu", "NuHat", "XLoc", "YLoc", "PointID",
	"UVel", "VVel", "DUDX", "DUDY", "DVDX", "DVDY", "WallDistance", "DNuHatDX",
	"DNuHatDY", "Production", "Destruction", "CrossProduction", "Fw", "OmegaBar",
	"OmegaNondimer", "Chi"}

var (
	Source       = mustFindStringIndex(features, "Source")
	PointID      = mustFindStringIndex(features, "PointID")
	UVel         = mustFindStringIndex(features, "UVel")
	VVel         = mustFindStringIndex(features, "VVel")
	Nu           = mustFindStringIndex(features, "Nu")
	NuHat        = mustFindStringIndex(features, "NuHat")
	WallDistance = mustFindStringIndex(features, "WallDistance")

	DNuHatDX        = mustFindStringIndex(features, "DNuHatDX")
	DNuHatDY        = mustFindStringIndex(features, "DNuHatDY")
	DUDX            = mustFindStringIndex(features, "DUDX")
	DUDY            = mustFindStringIndex(features, "DUDY")
	DVDX            = mustFindStringIndex(features, "DVDX")
	DVDY            = mustFindStringIndex(features, "DVDY")
	CrossProduction = mustFindStringIndex(features, "CrossProduction")
	Production      = mustFindStringIndex(features, "Production")
	Destruction     = mustFindStringIndex(features, "Destruction")
	Fw              = mustFindStringIndex(features, "Fw")
	Chi             = mustFindStringIndex(features, "Chi")
	OmegaBar        = mustFindStringIndex(features, "OmegaBar")
	OmegaNondimer   = mustFindStringIndex(features, "OmegaNondimer")
)

var estFeatures = []string{"DUDXEst", "DUDYEst", "DVDXEst", "DVDYEst", "DNuHatDXEst",
	"DNuHatDYEst", "NuSum", "NuSumDX", "NuSumDY"}

var (
	DUDXEst     = mustFindStringIndex(estFeatures, "DUDXEst")
	DUDYEst     = mustFindStringIndex(estFeatures, "DUDYEst")
	DVDXEst     = mustFindStringIndex(estFeatures, "DVDXEst")
	DVDYEst     = mustFindStringIndex(estFeatures, "DVDYEst")
	DNuHatDXEst = mustFindStringIndex(estFeatures, "DNuHatDXEst")
	DNuHatDYEst = mustFindStringIndex(estFeatures, "DNuHatDYEst")
	NuSum       = mustFindStringIndex(estFeatures, "NuSum")
	NuSumDX     = mustFindStringIndex(estFeatures, "NuSumDX")
	NuSumDY     = mustFindStringIndex(estFeatures, "NuSumDY")
)

var newFeatures = []string{"Computed_Source", "SourceEstDirect"}

var (
	ComputedSource  = mustFindStringIndex(newFeatures, "Computed_Source")
	SourceEstDirect = mustFindStringIndex(newFeatures, "SourceEstDirect")
)

func main() {
	nDim := 2
	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/med/Flatplate_Re_3e_06_test/turb_flatplate_sol.dat"
	//meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/mesh_flatplate_turb_137x97.su2"
	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/pressuregradient/med/Flatplate_Re_5e_06_Cp_0.3/turb_flatplate_sol.dat"
	// meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/mesh_flatplate_turb_137x97.su2"
	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012/Naca0012_0/solution_flow.dat"
	//meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012/ransuqbase/mesh_NACA0012_turb_897x257.su2"

	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012machine/solution_flow.dat"
	//meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012machine/mesh_NACA0012_turb_449x129.su2"

	dataset := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012machine/weighted_least_squares/solution_flow.dat"
	meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012machine/mesh_NACA0012_turb_449x129.su2"

	fmt.Println("Loading data")
	data := loadData(dataset, features)
	msh := loadMesh(meshfilename)
	fmt.Println("Done loading data")

	// Make sure that the node numbers match their index
	for i, pt := range data {
		// Don't just delete this. Parts of the code assume this invariant
		if int(pt[PointID]) != i {
			log.Fatal("point ID does not match")
		}
	}

	// Make sure the mesh and data agree on the size
	if len(data) != len(msh.Points) {
		fmt.Println("len data = ", len(data), " mesh elements = ", len(msh.Points))
		log.Fatal("num elements mismatch")

	}
	fmt.Println("Number of data points is ", len(data))

	// Get all the points for weighted least squares
	planePoints := make([][]*scattered.PointMV, len(data))
	for i := range planePoints {
		nNeighbors := len(msh.Points[i].OrderedNeighbors)
		planePoints[i] = make([]*scattered.PointMV, nNeighbors)
		for j := range msh.Points[i].OrderedNeighbors {
			dist := floats.Distance(msh.Points[i].Location, msh.Points[i].OrderedNeighbors[j].Location, 2)
			//invDistSq := 1.0 / (dist * dist)
			weight := 1.0 / dist
			planePoints[i][j] = &scattered.PointMV{
				Location: msh.Points[i].OrderedNeighbors[j].Location,
				Weight:   weight,
			}
		}
	}

	extraData := make([][]float64, len(data))
	newData := make([][]float64, len(data))
	for i := range extraData {
		extraData[i] = make([]float64, len(estFeatures))
		newData[i] = make([]float64, len(newFeatures))
	}

	var intercept scattered.Intercept
	deriv := make([]float64, nDim)
	// Do the estimates
	for i := range data {
		thisLocation := msh.Points[i].Location

		//fmt.Printf("%#v\n", msh.Points[i])

		// U velocity gradient estimate
		intercept = setPointValues(data, UVel, planePoints[i], msh.Points[i])
		scattered.Plane(thisLocation, planePoints[i], intercept, deriv)
		extraData[i][DUDXEst] = deriv[0]
		extraData[i][DUDYEst] = deriv[1]

		// V velocity gradient estimate
		intercept = setPointValues(data, VVel, planePoints[i], msh.Points[i])
		scattered.Plane(thisLocation, planePoints[i], intercept, deriv)
		extraData[i][DVDXEst] = deriv[0]
		extraData[i][DVDYEst] = deriv[1]

		// NuHat gradient estimate
		intercept = setPointValues(data, NuHat, planePoints[i], msh.Points[i])
		scattered.Plane(thisLocation, planePoints[i], intercept, deriv)
		extraData[i][DNuHatDXEst] = deriv[0]
		extraData[i][DNuHatDYEst] = deriv[1]

		nusum := data[i][Nu] + data[i][NuHat]
		extraData[i][NuSum] = nusum
		extraData[i][NuSumDX] = nusum * extraData[i][DNuHatDXEst]
		extraData[i][NuSumDY] = nusum * extraData[i][DNuHatDYEst]

		// Compute based on SA source
		saVars := &sa.SA{
			NDim:     2,
			Nu:       data[i][Nu],
			NuHat:    data[i][NuHat],
			DNuHatDX: []float64{extraData[i][DNuHatDXEst], extraData[i][DNuHatDYEst]},
			DUIdXJ: [][]float64{
				{extraData[i][DUDXEst], extraData[i][DVDXEst]},
				{extraData[i][DUDYEst], extraData[i][DVDYEst]},
			},
			WallDistance: data[i][WallDistance],
		}

		newData[i][SourceEstDirect] = saVars.Source()
		if data[i][WallDistance] < 1e-10 {
			newData[i][SourceEstDirect] = 0
		}
		// Check the estimate matched
		//if math.Abs(newData[i][SourceEstDirect]-data[i][Source]) > 1e-7 {
		if false {
			//if !floats.EqualWithinAbsOrRel(newData[i][SourceEstDirect], data[i][Source], 1e-3, 1e0) {
			//if false {
			fmt.Println("i = ", i)
			fmt.Println("real source = ", data[i][Source])
			fmt.Println("est  source = ", newData[i][SourceEstDirect])

			weights := make([]float64, len(planePoints[i]))
			for j := range weights {
				weights[j] = planePoints[i][j].Weight
			}
			fmt.Println("weights", weights)

			fmt.Println("locations = ")
			for j := 0; j < len(planePoints[i]); j++ {
				fmt.Println(planePoints[i][j].Location)
			}
			intercept = setPointValues(data, NuHat, planePoints[i], msh.Points[i])
			fmt.Println("nu hat values = ")
			for j := 0; j < len(planePoints[i]); j++ {
				fmt.Println(planePoints[i][j].Value)
			}

			fmt.Println("u deriv")
			fmt.Println(extraData[i][DUDXEst], extraData[i][DUDYEst])
			fmt.Println(data[i][DUDX], data[i][DUDY])

			fmt.Println("v deriv")
			fmt.Println(extraData[i][DVDXEst], extraData[i][DVDYEst])
			fmt.Println(data[i][DVDX], data[i][DVDY])

			fmt.Println("Nu hat deriv")
			fmt.Println(extraData[i][DNuHatDXEst], extraData[i][DNuHatDYEst])
			fmt.Println(data[i][DNuHatDX], data[i][DNuHatDY])

			log.Fatal("bad direct point in loop")
		}
	}

	// Compute dnuhatdx
	for i := range data {
		thisLocation := msh.Points[i].Location
		intercept = setPointValues(extraData, NuSumDX, planePoints[i], msh.Points[i])
		scattered.Plane(thisLocation, planePoints[i], intercept, deriv)
		dNusumdxDx := deriv[0]
		//dNusumdxDy := deriv[1]

		intercept = setPointValues(extraData, NuSumDY, planePoints[i], msh.Points[i])
		scattered.Plane(thisLocation, planePoints[i], intercept, deriv)
		//dNusumdyDx := deriv[0]
		dNusumdyDy := deriv[1]

		rhs := (dNusumdxDx + dNusumdyDy) / sa.Sigma

		lhs := data[i][UVel]*extraData[i][DNuHatDXEst] + data[i][VVel]*extraData[i][DNuHatDYEst]

		source := lhs - rhs
		newData[i][ComputedSource] = source

		if data[i][WallDistance] < 1e-10 {
			newData[i][ComputedSource] = 0
		}

		if math.IsNaN(source) {
			panic("nan value of source")
		}

		// if false {
		//if math.Abs(source-data[i][Source]) > 1e-2 {
		//if math.Abs(source-newData[i][SourceEstDirect]) > 1e-4 {
		if false {
			fmt.Println("Budget source: ", source)
			fmt.Println("Source est direct", newData[i][SourceEstDirect])
			fmt.Println("Real source: ", data[i][Source])
			log.Fatal("bad value of budget balanced source")
		}
	}

	// Make plots of real vs. est and real vs. computed

	err := makeplot(data, Source, newData, SourceEstDirect, "comp_direct.pdf", "Real", "EstDirect")
	if err != nil {
		log.Fatal(err)
	}
	err = makeplot(data, Source, newData, ComputedSource, "comp_budget.pdf", "Real", "EstBudget")
	if err != nil {
		log.Fatal(err)
	}

	log.Fatal("Need to code ability to save data -- try with exact SA NACA 0012 simulation")
}

func makeplot(data1 [][]float64, col1 int, data2 [][]float64, col2 int, filename, xLabel, yLabel string) error {
	p, err := plot.New()
	if err != nil {
		return err
	}
	p.X.Label.Text = xLabel
	p.Y.Label.Text = yLabel

	pts := make(plotter.XYs, len(data1))
	for i := 0; i < len(data1); i++ {
		pts[i].X = data1[i][col1]
		pts[i].Y = data2[i][col2]
	}

	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return err
	}
	p.Add(scatter)
	/*
		err = plotutil.AddLinePoints(p, "Comp", pts)
		if err != nil {
			return err
		}
	*/
	err = p.Save(4, 4, filename)
	if err != nil {
		return err
	}
	return nil
}

func setPointValues(data [][]float64, dataIdx int, points []*scattered.PointMV, point *mesh.Point) scattered.Intercept {
	for i, neighbor := range point.OrderedNeighbors {
		idx := neighbor.Id
		points[i].Value = data[idx][dataIdx]
	}
	return scattered.Intercept{
		Force: true,
		Value: data[point.Id][dataIdx],
	}
}

func mustFindStringIndex(strs []string, str string) int {
	for i, v := range strs {
		if v == str {
			return i
		}
	}
	log.Fatal("string " + str + " not found")
	return -1
}

func loadMesh(meshfilename string) *mesh.SU2 {
	// Load in the mesh file so there is connectivity information
	f, err := os.Open(meshfilename)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	msh := &mesh.SU2{}
	_, err = msh.ReadFrom(f)
	if err != nil {
		log.Fatal(err)
	}
	return msh
}

func loadData(dataset string, features []string) [][]float64 {
	set := &dataloader.Dataset{
		Name:     "Data",
		Filename: dataset,
		Format:   &dataloader.SU2_restart_2dturb{},
	}

	allData, err := dataloader.Load(features, []*dataloader.Dataset{set})
	if err != nil {
		log.Fatal(err)
	}
	data := allData[0] // just limit it to this dataset
	return data
}
