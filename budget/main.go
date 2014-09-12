package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"code.google.com/p/plotinum/plot"
	"code.google.com/p/plotinum/plotter"
	"code.google.com/p/plotinum/plotutil"

	"github.com/btracey/numcsv"
	"github.com/btracey/ransuq/dataloader"
	"github.com/btracey/su2tools/mesh"
	"github.com/btracey/turbulence/sa"
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

const (
	kappa = 0.41
	cb1   = 0.1355
	sigma = 2.0 / 3.0
	cb2   = 0.622
	cw2   = 0.3
	cw3   = 2.0
	cw3_6 = cw3 * cw3 * cw3 * cw3 * cw3 * cw3
	cv1   = 7.1
	ct3   = 1.2
	ct4   = 0.5
	cw1   = cb1/kappa*kappa + (1+cb2)/(sigma)
)

func main() {
	tinit := time.Now()
	t := time.Now()
	fmt.Println("Start")

	printVals := true

	//defer profile.Start(profile.CPUProfile).Stop()

	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/med/Flatplate_Re_7e_06/turb_flatplate_sol.dat"
	// meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/mesh_flatplate_turb_137x97.su2"
	//dataset := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/pressuregradient/med/Flatplate_Re_5e_06_Cp_0.3/turb_flatplate_sol.dat"
	// meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/flatplate/mesh_flatplate_turb_137x97.su2"
	dataset := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012/Naca0012_0/solution_flow.dat"
	meshfilename := "/Users/brendan/Documents/mygo/data/ransuq/airfoil/naca0012/ransuqbase/mesh_NACA0012_turb_897x257.su2"

	csvComma := "\t"

	// Data to load from SU2 Data
	features := []string{"Source", "Nu", "NuHat", "XLoc", "YLoc", "PointID",
		"UVel", "VVel", "DUDX", "DUDY", "DVDX", "DVDY", "WallDistance", "DNuHatDX",
		"DNuHatDY", "Production", "Destruction", "CrossProduction", "Fw", "OmegaBar",
		"OmegaNondimer", "Chi"}

	// Find which columns for needed data
	sourceIdx := findStringLocation(features, "Source")
	pointIdx := findStringLocation(features, "PointID")
	uvelIdx := findStringLocation(features, "UVel")
	vvelIdx := findStringLocation(features, "VVel")
	nuIdx := findStringLocation(features, "Nu")
	nuHatIdx := findStringLocation(features, "NuHat")
	wallDistIdx := findStringLocation(features, "WallDistance")

	dNuHatDXIdx := findStringLocation(features, "DNuHatDX")
	dNuHatDYIdx := findStringLocation(features, "DNuHatDY")
	dudxIdx := findStringLocation(features, "DUDX")
	dudyIdx := findStringLocation(features, "DUDY")
	dvdxIdx := findStringLocation(features, "DVDX")
	dvdyIdx := findStringLocation(features, "DVDY")
	crossProdIdx := findStringLocation(features, "CrossProduction")
	prodIdx := findStringLocation(features, "Production")
	destIdx := findStringLocation(features, "Destruction")
	fwIdx := findStringLocation(features, "Fw")
	chiIdx := findStringLocation(features, "Chi")
	omegaBarIdx := findStringLocation(features, "OmegaBar")
	omegaNondimerIdx := findStringLocation(features, "OmegaNondimer")
	//residIdx := findStringLocation(features, "Residual_3")

	//yLocIdx := findStringLocation(features, "YLoc")

	data := loadData(dataset, features)

	msh := loadMesh(meshfilename)
	fmt.Println("Load time = ", time.Since(t))
	t = time.Now()

	if len(data) != len(msh.Points) {
		log.Fatal("mismatch in number of data points and number of mesh points")
	}
	// Find the mapping from SU2 point IDS to rows in the data matrix
	idxPointToData := make(map[mesh.PointID]int)
	idxDataToPoint := make(map[int]mesh.PointID)
	for i, pt := range data {
		id := pt[pointIdx]
		idxPointToData[mesh.PointID(id)] = i
		idxDataToPoint[i] = mesh.PointID(id)
	}

	//locations := getNeighborLocations(msh)
	quantityIdxs := []int{uvelIdx, vvelIdx, nuIdx, nuHatIdx}
	//quantityIdxs := []int{nuHatIdx}
	derivatives := computeDerivatives(msh, data, quantityIdxs, idxPointToData)

	//neighbors := getNeighborLocs(msh)

	// Compute first derivatives at all the points

	nuHatDerivIdx := findIntLocation(quantityIdxs, nuHatIdx)
	uVelDerivIdx := findIntLocation(quantityIdxs, uvelIdx)
	vVelDerivIdx := findIntLocation(quantityIdxs, vvelIdx)

	/*
		derivatives := make(map[mesh.PointID][][]float64)
		for k := 0; k < len(msh.Points); k++ {

			// Try to compute the budget
			id := msh.Points[k].Id

			nNeighbors := len(msh.Points[id].Neighbors)
			neighborLocs := make([][]float64, nNeighbors)
			neigborValues := make([][]float64, nNeighbors)
			for i := 0; i < nNeighbors; i++ {
				neigborValues[i] = make([]float64, len(quantityIdxs))
			}
			count := 0
			for pointID := range msh.Points[id].Neighbors {
				neighborLocs[count] = msh.Points[pointID].Location
				idx, ok := idxPointToData[pointID]
				if !ok {
					panic(fmt.Sprintf("point %d not found", pointID))
				}
				for j := 0; j < len(quantityIdxs); j++ {
					neigborValues[count][j] = data[idx][quantityIdxs[j]]
				}
				count++
			}

			thisIdx, ok := idxPointToData[id]
			if !ok {
				panic("this point not known")
			}
			thisVal := make([]float64, len(quantityIdxs))
			for j := 0; j < len(quantityIdxs); j++ {
				thisVal[j] = data[thisIdx][quantityIdxs[j]]
			}
			derivatives[id] = estimateGradients(msh.Points[id].Location, thisVal, neighborLocs, neigborValues)
			//fmt.Println(derivatives[k])
		}
	*/

	// Form (v + vhat) dnuhat/dx
	nuNuhatDxy := make([][]float64, len(data))
	for i := range nuNuhatDxy {
		nuNuhatDxy[i] = make([]float64, 2)
	}
	//nuNuhatDDy := make([]float64, len(data))
	for k := 0; k < len(msh.Points); k++ {
		id := msh.Points[k].Id
		thisIdx := idxPointToData[id]
		nu := data[thisIdx][nuIdx]
		nuhat := data[thisIdx][nuHatIdx]
		dNuHatDx := derivatives[id][nuHatDerivIdx][0]
		dNuHatDy := derivatives[id][nuHatDerivIdx][1]

		//fmt.Printf("DNuHatDX est = %.6e, \t real is %.6e\n", dNuHatDx, data[thisIdx][dNuHatDxIdx])
		//fmt.Printf("DNuHatDY est = %.6e, \t real is %.6e\n", dNuHatDy, data[thisIdx][dNuHatDyIdx])

		nuNuhatDxy[thisIdx][0] = (nu + nuhat) * dNuHatDx
		nuNuhatDxy[thisIdx][1] = (nu + nuhat) * dNuHatDy
		/*
			if k == 17145 {
				fmt.Println("dNuHatDX = ", derivatives[id][nuHatDerivIdx])
				fmt.Println("nuNuhatDXY = ", nuNuhatDxy[thisIdx])
				fmt.Println("this idx = ", thisIdx, " id = ", id)
				os.Exit(1)
			}
		*/
	}
	//os.Exit(1)

	// Need to compute derivatives of (v + nuhat)dnuhat/dxj
	nuNuhatDerivs := computeDerivatives(msh, nuNuhatDxy, []int{0, 1}, idxPointToData)

	//fmt.Println(nuNuhatDerivs[17145])
	//os.Exit(1)

	/*
		// Now compute those second derivatives
		nuNuhatXDeriv := make(map[mesh.PointID]float64) // d/dxj ((v + vhat) * dnuhat/dxj)
		nuNuhatYDeriv := make(map[mesh.PointID]float64)
		for k := 0; k < len(msh.Points); k++ {
			nNeighbors := len(msh.Points[id].NeighborIDs)
			neighborLocs := make([][]float64, nNeighbors)
			neigborValues := make([][]float64, nNeighbors)
			for i := 0; i < nNeighbors; i++ {
				neigborValues[i] = make([]float64, 2)
			}
		}
	*/
	//fmt.Println(nuNuhatDDx, nuNuhatDDy)

	/*
		for k := 0; k < len(msh.Points); k++ {
			id := msh.Points[k].Id
			idx := idxPointToData[id]
			fmt.Printf("dudx est: %.6g\tdudx SU2: %.6g\n", derivatives[id][0][1], data[idx][dudxIdx])
		}
	*/

	// approximate source term
	sourceEst := make([]float64, len(data))

	sourceEstDirect := make([]float64, len(data))

	for k := 0; k < len(msh.Points); k++ {
		id := msh.Points[k].Id
		idx := idxPointToData[id]

		uvec := []float64{data[idx][uvelIdx], data[idx][vvelIdx]}
		nuHatDeriv := derivatives[id][nuHatDerivIdx]
		lhs := floats.Dot(uvec, nuHatDeriv)
		rhs := (1.0 / sigma) * (nuNuhatDerivs[id][0][0] + nuNuhatDerivs[id][1][1])

		sourceEst[idx] = lhs - rhs

		// HACK!!!!!!!!!
		if data[idx][wallDistIdx] < 1e-10 {
			sourceEst[idx] = 0
		}

		// Do a double check of the source with the real and fake derivatives... see if there's a consistency issue

		// Compute based on the SA source equation directly
		saVars := sa.SA{
			NDim:     2,
			Nu:       data[idx][nuIdx],
			NuHat:    data[idx][nuHatIdx],
			DNuHatDX: derivatives[id][nuHatDerivIdx],
			DUIdXJ: [][]float64{
				derivatives[id][uVelDerivIdx],
				derivatives[id][vVelDerivIdx],
			},
			WallDistance: data[idx][wallDistIdx],
		}

		sourceEstDirect[idx] = saVars.Source()

		if printVals {
			fmt.Println()
			fmt.Println(data[idx][sourceIdx], sourceEst[idx], sourceEstDirect[idx])
			fmt.Println("uvec = ", uvec)
			fmt.Println("nhatderiv = ", nuHatDeriv)
			fmt.Println("lhs = ", lhs)
			fmt.Println("rhs = ", rhs)
			//fmt.Println(msh.Points[k].Location[1])
			//fmt.Println(data[idx][wallDistIdx])
			fmt.Println("NuHat = ", saVars.NuHat, data[idx][nuHatIdx])
			fmt.Println("Chi = ", saVars.Chi, data[idx][chiIdx])
			fmt.Println("WallDist = ", saVars.WallDistance, data[idx][wallDistIdx])
			fmt.Println("u deriv", derivatives[id][uVelDerivIdx])
			fmt.Println("real u deriv", data[idx][dudxIdx], data[idx][dudyIdx])
			fmt.Println("v deriv", derivatives[id][vVelDerivIdx])
			fmt.Println("real v deriv", data[idx][dvdxIdx], data[idx][dvdyIdx])
			fmt.Println("est = ", derivatives[id][nuHatDerivIdx])
			fmt.Println("real = ", data[idx][dNuHatDXIdx], data[idx][dNuHatDYIdx])
			fmt.Println("Cross Production = ", saVars.CrossProduction, data[idx][crossProdIdx])
			omegaNondimer := (saVars.Nu + saVars.NuHat) / (saVars.WallDistance * saVars.WallDistance)
			fmt.Println("OmegaNondim", omegaNondimer, data[idx][omegaNondimerIdx])
			fmt.Println("OmegaBar", saVars.Omega/omegaNondimer, data[idx][omegaBarIdx])
			fmt.Println("Production = ", saVars.Production, data[idx][prodIdx])
			fmt.Println("Fw = ", saVars.Fw, data[idx][fwIdx])
			fmt.Println("Destruction = ", saVars.Destruction, data[idx][destIdx])
			fmt.Println("id = ", id)
		}

		if math.Abs(sourceEstDirect[idx]-data[idx][sourceIdx]) > 1e-9 {
			fmt.Println(sourceEstDirect[idx], data[idx][sourceIdx], sourceEstDirect[idx]-data[idx][sourceIdx])
			log.Fatal("bad direct point in loop")
		}
		if math.Abs(sourceEst[idx]-data[idx][sourceIdx]) > 1e1 {
			fmt.Println(data[idx][sourceIdx], sourceEst[idx], sourceEstDirect[idx])
			fmt.Println(sourceEst[idx], data[idx][sourceIdx], sourceEst[idx]-data[idx][sourceIdx])
			log.Fatal("egregious error in sourceEst")
		}
		if math.IsNaN(sourceEstDirect[idx]) {
			panic("nan value")
		}

		/*
			dudxTensor := [][]float64{derivatives[id][uVelDerivIdx], derivatives[id][vVelDerivIdx]}
			var vorticity float64
			for i := 0; i < 2; i++ {
				for j := 0; j < 2; j++ {
					diff := dudxTensor[i][j] - dudxTensor[j][i]
					vorticity += diff * diff
				}
			}
			vorticity = math.Sqrt(2 * vorticity)
			nuhat := data[idx][nuHatIdx]
			nu := data[idx][nuIdx]
			wallDist := data[idx][wallDistIdx]

			chi := nuhat / nu
			fv1 := chi * chi * chi / (chi*chi*chi + cv1*cv1*cv1)
			fv2 := 1 - chi/(1+chi*fv1)
			Shat := vorticity + (nuhat*nuhat)/(kappa*kappa*wallDist*wallDist)*fv2

			ft2 := ct3 * math.Exp(-ct4*chi*chi)
			prod := cb1 * (1 - ft2) * Shat * nuhat

			r := math.Min(nuhat/(Shat*kappa*kappa*wallDist*wallDist), 10)
			g := r + cw2*(math.Pow(r, 6)-r)
			fw := g * math.Pow((1+cw3_6)/(math.Pow(g, 6)+cw3_6), 1.0/6.0)

			dest := (cw1*fw - cb1/(kappa*kappa)*ft2) * (nuhat * nuhat / (wallDist * wallDist))

			diff := nuNuhatDerivs[id][0][0] + nuNuhatDerivs[id][1][1]
			var nuHatDerivMag float64
			for i := 0; i < 2; i++ {
				nuHatDerivMag += nuHatDeriv[i] * nuHatDeriv[i]
			}
			crosprod := 1 / sigma * (diff + cb2*nuHatDerivMag)

			sourceEst[idx] = prod - dest + crosprod
		*/
	}

	fmt.Println("Compute time = ", time.Since(t))
	t = time.Now()
	// Make a plot of the budget comparison
	pts := make(plotter.XYs, len(sourceEst))

	for k := 0; k < len(msh.Points); k++ {
		//fmt.Println(sourceEst[k], "\t", data[k][sourceIdx], "\t", sourceEst[k]-data[k][sourceIdx])
		pts[k].X = data[k][sourceIdx]
		pts[k].Y = sourceEst[k]
		//pts[k].Y = sourceEstDirect[k]

		//		fmt.Println(pts[k])
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	//line := plotter.NewFunction(func(x float64) float64 { return x })
	//p.Add(line)
	err = plotutil.AddLinePoints(p, "Comp", pts)
	if err != nil {
		panic(err)
	}
	if err := p.Save(4, 4, "budgetplot.pdf"); err != nil {
		log.Fatal(err)
	}
	fmt.Println("Plot time = ", time.Since(t))
	t = time.Now()

	// Now that have the source estimate, need to rewrite the file.

	// Load all the data from the datafile
	f, err := os.Open(dataset)
	if err != nil {
		log.Fatal(err)
	}

	csvreader := numcsv.NewReader(f)
	csvreader.Comma = csvComma

	head, err := csvreader.ReadHeading()
	if err != nil {
		fmt.Println("Error in ReadHeading")
		log.Fatal(err)
	}
	mat, err := csvreader.ReadAll()
	if err != nil {
		fmt.Println("Error in ReadAll")
		log.Fatal(err)
	}
	// Need to append the extra source values onto the end
	head = append(head, "Computed_Source")

	r, c := mat.Dims()
	newmat := mat64.NewDense(r, c+1, nil)
	cr, cc := newmat.Copy(mat)

	//r2, c2 := newmat.Dims()

	//fmt.Println("orig dims = ", r, c)
	//fmt.Println("newmat dims = ", r2, c2)
	if cr != r {
		panic("wrong number of rows copied")
	}
	if cc != c {
		panic("wrong number of columns copied")
	}

	ptIdx := findStringLocation(head, "PointID")

	for i := 0; i < r; i++ {
		id := newmat.At(i, ptIdx)
		srcIdx := idxPointToData[mesh.PointID(id)]
		src := sourceEst[srcIdx]
		newmat.Set(i, c, src)
	}

	ext := filepath.Ext(dataset)
	pre := dataset[:len(dataset)-len(ext)]
	newFilename := pre + "_budget" + ext

	fnew, err := os.Create(newFilename)
	defer fnew.Close()

	csvWriter := numcsv.NewWriter(fnew)
	csvWriter.QuoteHeading = true
	csvWriter.Comma = csvComma
	err = csvWriter.WriteAll(head, newmat)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Final Save time = ", time.Since(t))
	fmt.Println("Full time: ", time.Since(tinit))
}

// Compute the derivatives of the indexes of data
func computeDerivatives(msh *mesh.SU2, data [][]float64, idxs []int, idxPointToData map[mesh.PointID]int) map[mesh.PointID][][]float64 {
	// Keep taking points
	nVars := len(idxs)
	thisValue := make([]float64, nVars)

	derivatives := make(map[mesh.PointID][][]float64, len(msh.Points))

	for k := 0; k < len(msh.Points); k++ {
		pt := msh.Points[k]
		// Get all the neighbor values
		nNeighbors := len(pt.Neighbors)
		neighborValues := make([][]float64, nNeighbors)
		for i := 0; i < nNeighbors; i++ {
			neighborValues[i] = make([]float64, nVars)
		}
		neighborLocation := make([][]float64, nNeighbors)
		// add the values
		i := 0
		for _, neighbor := range pt.Neighbors {
			id := neighbor.Id
			values := data[idxPointToData[id]]
			for j := range neighborValues[i] {
				neighborValues[i][j] = values[idxs[j]]
			}
			neighborLocation[i] = msh.Points[id].Location
			i++
		}
		thisIdx := idxPointToData[pt.Id]
		for i := range thisValue {
			thisValue[i] = data[thisIdx][idxs[i]]
		}
		/*
			if k == 0 {
				fmt.Println()
				fmt.Println("pt loc", pt.Location)
				fmt.Println("pt neighbors = ")
				for k := range pt.Neighbors {
					fmt.Println(k)
				}
				fmt.Println("neighbor loc ", neighborLocation)
				fmt.Println("this value", thisValue)
				fmt.Println("neighbor Value", neighborValues)
			}
		*/

		derivatives[pt.Id] = estimateGradients(pt.Location, thisValue, neighborLocation, neighborValues)

		fmt.Println("firstGlobal = ", firstGlobal, "k = ", k)
		if k == 17145 {
			if !firstGlobal {
				firstGlobal = true // This is a hack to show the nu derivatives
			} else {
				fmt.Println("deriv = ", derivatives[pt.Id])
				//			fmt.Println("real = ", data[k][8], data[k][9])
				//fmt.Println("wall dist = ", data[k][10])
				os.Exit(1)
			}
		}

	}
	return derivatives
}

var firstGlobal bool

type Neighbor struct {
	Id       mesh.PointID
	Location []float64
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

func loadMesh(meshfilename string) *mesh.SU2 {
	// Load in the mesh file so there is connectivity information
	f, err := os.Open(meshfilename)
	if err != nil {
		log.Fatal(err)
	}

	msh := &mesh.SU2{}
	_, err = msh.ReadFrom(f)
	if err != nil {
		log.Fatal(err)
	}
	return msh
}

func findStringLocation(s []string, str string) int {
	for i, v := range s {
		if v == str {
			return i
		}
	}
	return -1
}

func findIntLocation(s []int, idx int) int {
	for i, v := range s {
		if v == idx {
			return i
		}
	}
	return -1
}

// estimate gradient
func estimateGradients(pointLocation []float64, pointValue []float64,
	altLocations [][]float64, altValues [][]float64) (derivative [][]float64) {
	// compute all the distances
	fmt.Println()
	fmt.Println("point value =", pointValue)
	fmt.Println("neighbor values", altValues)
	fmt.Println("point location = ", pointLocation)
	fmt.Println("alt locations = ", altLocations)

	dists := make([]float64, len(altLocations))
	tmp := make([]float64, len(pointLocation))
	for i := range altLocations {
		floats.SubTo(tmp, pointLocation, altLocations[i])
		dists[i] = floats.Norm(tmp, 2)
	}

	weights := make([]float64, len(altLocations))
	// Weights are proportional to that norm
	for i, v := range dists {
		weights[i] = math.Sqrt(1 / (v * v))
	}

	// Make the weight at the current location the same as the maximum weight
	// so that it scales well with mesh size
	mv, _ := floats.Max(weights)

	weights = append(weights, mv)
	/*
		weights := make([]float64, len(altLocations)+1)
		for i := range weights {
			weights[i] = 1
		}
	*/

	//fmt.Println("dists = ", dists)
	//fmt.Println("weights = ", weights)

	// For numerical stability reasons, need to make go through the (0, 0) --> 0
	// point. Translate such that this point is at 0,0
	// Want to fix A_-1 * 1 A_0 x_0 + A_1 x_1 + ... = value
	nDim := len(pointLocation)
	nSamples := len(altValues)
	nOutputs := len(altValues[0])

	//A := mat64.NewDense(nSamples+1, nDim+1, nil)
	A := mat64.NewDense(nSamples, nDim, nil)
	for i := 0; i < nSamples; i++ {
		/*
			A.Set(i, 0, 1)
			for j := 0; j < nDim; j++ {
				A.Set(i, j+1, altLocations[i][j])
			}
		*/
		for j := 0; j < nDim; j++ {
			A.Set(i, j, altLocations[i][j]-pointLocation[j])
		}
	}
	/*
		// Set the local point
		A.Set(nSamples, 0, 1)
		for j := 0; j < nDim; j++ {
			A.Set(nSamples, j+1, pointLocation[j])
		}
	*/

	//b := mat64.NewDense(nSamples+1, nOutputs, nil)
	b := mat64.NewDense(nSamples, nOutputs, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nOutputs; j++ {
			b.Set(i, j, altValues[i][j]-pointValue[j])
		}
	}
	/*
		for j := 0; j < nOutputs; j++ {
			b.Set(nSamples, j, pointValue[j])
		}
	*/

	fmt.Println("A unweighted is ", A)
	fmt.Println("b unweighted is ", b)
	x := mat64.Solve(A, b)
	fmt.Println("x unweighted = ", x)

	// Scale A and b by the weights
	for i := 0; i < nSamples; i++ {
		//for j := 0; j < nDim+1; j++ {
		for j := 0; j < nDim; j++ {
			v := A.At(i, j) * weights[i]
			A.Set(i, j, v)
		}
		for j := 0; j < nOutputs; j++ {
			v := b.At(i, j) * weights[i]
			b.Set(i, j, v)
		}
	}

	// Solve
	x = mat64.Solve(A, b)

	fmt.Println("A is ", A)
	fmt.Println("b is ", b)
	fmt.Println("x weighted = ", x)

	// Pull out the derivative
	derivative = make([][]float64, nOutputs)
	for i := 0; i < nOutputs; i++ {
		derivative[i] = make([]float64, len(pointLocation))
		for j := 0; j < len(pointLocation); j++ {
			//derivative[i][j] = x.At(j+1, i)
			derivative[i][j] = x.At(j, i)
		}
	}
	return derivative
}

/*
// knn finds the k points closest to the x, y data specified
func knn(k int, x, y float64, xdata, ydata []float64) []int {
	dists := make([]float64, len(xdata))
	inds := make([]int, len(xdata))
	for i := range xdata {
		dists[i] = (x-xdata[i])*(x-xdata[i]) + (y-ydata[i])*(y-ydata[i])
	}
	floats.Sort
}
*/
