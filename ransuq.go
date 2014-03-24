package ransuq

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
)

type GenerationError struct {
	Training []error
	Testing  []error
}

func (g GenerationError) Error() string {
	var str string
	if g.Training != nil {
		str += "training: "
		for i := range g.Training {
			str += fmt.Sprintf("error generating ", i, ": ", g.Training[i].Error())
		}
	}
	if g.Testing != nil {
		str += "testing: "
		for i := range g.Testing {
			str += fmt.Sprintf("error generating ", i, ": ", g.Testing[i].Error())
		}
	}
	return str
}

type Outscheduler struct {
	physicalRunChan chan Generatable
}

type generatableRunner struct {
	Generatables []Generatable
}

type GenerateFinished struct {
	Generatable
	Err error
}

type DatasetFinished struct {
	Dataset
	Err error
}

type DatasetRunner struct {
	compute chan Generatable
	done    chan GenerateFinished

	norun chan Dataset // For datasets that don't need to be run off externally
}

func NewDatasetRunner(compute chan Generatable, done chan GenerateFinished) *DatasetRunner {
	return &DatasetRunner{
		compute: compute,
		done:    done,
		norun:   make(chan Dataset),
	}
}

// Send sends the dataset out to be computed. Blocks until the send can happen
func (d *DatasetRunner) Compute(dataset Dataset) {
	// First, check if the data is actually a generatable
	generatable, ok := dataset.(Generatable)
	if !ok {
		id := dataset.ID()
		// Data was not a generatable. Record that that was so, and send it
		// as completed
		log.Printf("%v is not a generatable. Sending completion notice.", id)
		d.norun <- dataset
	}
	// It is a generatable, but it may be have already been generated. If so,
	// mark it as completed
	if generatable.Generated() {
		id := dataset.ID()
		log.Printf("%v has already been generated. Sending completion notice", id)
		d.norun <- dataset
	}
	// Otherwise, send it to be generated
	d.compute <- generatable
}

// Done returns a dataset that is completed. Blocks until a dataset has been completed
func (d *DatasetRunner) Done() DatasetFinished {
	select {
	case data := <-d.norun:
		return DatasetFinished{
			Dataset: data,
			Err:     nil,
		}
	case run := <-d.done:
		return DatasetFinished{
			Dataset: run.Generatable.(Dataset),
			Err:     run.Err,
		}
	}
}

/*
// trueDatasetGenerator sends out all the data sets to be run
type trueDatasetRunner struct {
	Datasets []Dataset
}

func (t *trueDatasetGenerator) Send(c chan Run) {
	for i, dataset := range t.Datasets {
		// First, check if the data is actually a generatable
		generatable, ok := runner.Set.(Generatable)
		if !ok {
			id := runner.Set.ID()
			// Data was not a generatable. Record that that was so, and send it
			// as completed
			log.Printf("%v is not a generatable. Sending completion notice.", runner.Set.ID())
			doneChan <- genComplete{ID: id}
		}
		// It is a generatable, but it may be have already been generated. If so,
		// mark it as completed
		if generatable.Generated() {
			id := runner.Set.ID()
			log.Printf("%v has already been generated. Sending completion notice", runner.Set.ID())
			doneChan <- genComplete{ID: id}
		}
		// Otherwise, send it to be generated
		outChan <- generatable
	}
	// All of the training and testing data was sent, so close the channel
	close(outChan)
}

func (d *trueDatasetGenerator) Receive(mlRunChan) {

}
*/

type mlRunData struct {
	*Settings
	trainError []error
}

// MultiTurb runs a list of cases
// TODO: Needs to be some form of cluster calls
func MultiTurb(runs []*Settings) error {
	// TODO: Set up a set of worker goroutines. (probably should be another function call)
	// the workers will need access to this channel

	// Need to hide a layer of abstraction here. Want the workers to read SU^2 runs
	// and ML Training, but also want to be able to control when specific pieces have been done

	// Problem is that we want to be able to send out postprocessing runs, training
	// runs, and testing runs

	/*
		datasetOutChan := make(chan Something)
		datasetDoneChan := make(chan Something)
		mlOutChan := make(chan SomethingElse)
		mlDoneChan := make(chan SomethingElse)

		runningDataChan := make(chan Generatable)
		dataFinishedChan := make(chan genComplete)

	*/

	//mlRunChan := make(chan mlRun)
	//learnerDoneChan := make(chan learner)

	// First, go through all of the training and testing sets, and get all of
	// the unique data

	// TODO: need to implement the correct channels
	physicalDataCompute := make(chan Generatable)
	physicalDataDone := make(chan GenerateFinished)

	// Get the unique data
	idToIdx, uniqueDatasets, isTraining, learners := uniqueDatasets(runs)
	_ = idToIdx
	_ = isTraining

	// Compute all of the physical data
	datasetRunner := NewDatasetRunner(physicalDataCompute, physicalDataDone)

	for _, datataset := range uniqueDatasets {
		go func(dataset Dataset) {
			datasetRunner.Compute(dataset)
		}(datataset)
	}

	mlChan := make(chan mlRunData)
	allDoneChan := make(chan *learner)
	// Launch goroutine to read in training data as they come in and
	go func() {
		mlStarted := make([]bool, len(learners))

		stillRunning := make(map[int]struct{})
		for i := range learners {
			stillRunning[i] = struct{}{}
		}

		for i := 0; i < len(uniqueDatasets); i++ {
			run := datasetRunner.Done()
			for idx := range stillRunning {
				learners[idx].RegisterCompletion(run)

				if !mlStarted[idx] && learners[idx].AllTrainDone() {
					mlChan <- mlRunData{learners[idx].Settings, learners[idx].trainErrs}
					mlStarted[idx] = true
				}
				if learners[idx].AllTrainDone() && learners[idx].AllTestDone() {
					allDoneChan <- learners[idx]
					delete(stillRunning, idx)
				}
			}
		}
		if len(stillRunning) != 0 {
			panic("Cases still running after data generated")
		}
		close(mlChan)
		close(allDoneChan)
		return
	}()

	//trainingCompleteChan := make(chan mlRunData)

	doneTraining := make(chan bool)
	doneGenTesting := make(chan bool)
	// Read the ml runs that are coming in, and launch the training code
	go func() {
		// This could be replaced with some worker
		for _ = range mlChan {
			ml := <-mlChan
			// Check if there were any errors
			fmt.Println("ml = ", ml)
			// Send the data on the done chan
		}
		doneTraining <- true
	}()

	// Read in the data learners when all the testing data has been done
	go func() {
		for _ = range allDoneChan {
			<-allDoneChan
		}
		doneGenTesting <- true
	}()

	<-doneTraining
	<-doneGenTesting
	fmt.Println("All done")
	return nil
}

type mlRun struct {
	Idx        int
	TraningErr []error
}

type genComplete struct {
	Idx int
	ID  string
	err error
}

/*
func sendRunners(runners []*datasetRunner, outChan chan Generatable, doneChan chan genComplete) {
	for i, runner := range runners {
		// First, check if the data is actually a generatable
		generatable, ok := runner.Set.(Generatable)
		if !ok {
			id := runner.Set.ID()
			// Data was not a generatable. Record that that was so, and send it
			// as completed
			log.Printf("%v is not a generatable. Sending completion notice.", runner.Set.ID())
			doneChan <- genComplete{ID: id}
		}
		// It is a generatable, but it may be have already been generated. If so,
		// mark it as completed
		if generatable.Generated() {
			id := runner.Set.ID()
			log.Printf("%v has already been generated. Sending completion notice", runner.Set.ID())
			doneChan <- genComplete{ID: id}
		}
		// Otherwise, send it to be generated
		outChan <- generatable
	}
	// All of the training and testing data was sent, so close the channel
	close(outChan)
}
*/

type datasetRunner struct {
	Set        Dataset
	Idx        int
	RunErr     error
	LoadErr    error
	IsTraining bool
}

// learner controls
type learner struct {
	*Settings
	trainRunning map[string]int // to local index
	trainIdx     []int          // map from local index to unique index
	trainErrs    []error

	testIdx     []int
	testRunning map[string]int
	testErrs    []error
}

// RegisterCompletion logs in the learner that that dataset has finished running
func (l *learner) RegisterCompletion(run DatasetFinished) {
	idx, ok := l.trainRunning[run.ID()]
	if ok {
		l.trainErrs[idx] = run.Err
		delete(l.trainRunning, run.ID())
	}

	idx, ok = l.testRunning[run.ID()]
	if ok {
		l.testErrs[idx] = run.Err
		delete(l.testRunning, run.ID())
	}
}

func (l *learner) AllTrainDone() bool {
	return len(l.trainRunning) == 0
}

func (l *learner) AllTestDone() bool {
	return len(l.testRunning) == 0
}

func uniqueDatasets(runs []*Settings) (m map[string]int, uniqueData []Dataset, isTraining map[string]bool, learners []*learner) {
	m = make(map[string]int)
	isTraining = make(map[string]bool)
	uniqueData = make([]Dataset, 0)
	learners = make([]*learner, len(runs))
	var uniqueIdx int
	for i, setting := range runs {
		learners[i].Settings = setting
		learners[i].trainRunning = make(map[string]int)
		learners[i].trainIdx = make([]int, len(setting.TrainingData))
		for j, dataset := range setting.TrainingData {
			idx, ok := m[dataset.ID()]
			if !ok {
				// Dataset doesn't exist yet, add it to the map
				uniqueIdx++
				m[dataset.ID()] = uniqueIdx
				uniqueData = append(uniqueData, dataset)
				idx = uniqueIdx
			}
			learners[i].trainIdx[j] = idx
			learners[i].trainRunning[dataset.ID()] = j
			isTraining[dataset.ID()] = true
		}
		for j, dataset := range setting.TestingData {
			idx, ok := m[dataset.ID()]
			if !ok {
				uniqueIdx++
				m[dataset.ID()] = uniqueIdx
				uniqueData = append(uniqueData, dataset)
				idx = uniqueIdx
			}
			learners[i].testIdx[j] = idx
			learners[i].testRunning[dataset.ID()] = j
		}
	}
	return
}

/*
func uniqueDatasets(runs []*Settings) (map[string]int, []*datasetRunner, []*learner) {
	m := make(map[string]int)
	uniqueData := make([]*datasetRunner, 0)
	learner := make([]*learner, len(runs))

	var uniqueIdx = 0
	for i, setting := range runs {
		learner[i].trainIdxs = make([]int, len(setting.TrainingData))
		learner[i].trainRunning = make(map[string]bool)
		for j, dataset := range setting.TrainingData {
			id := dataset.ID()
			idx, ok := m[dataset.ID()]
			if !ok {
				uniqueIdx = addUniqueDataset(m, uniqueData, uniqueIdx, dataset)
				idx = uniqueIdx
			}
			learner[i].trainIdxs[j] = idx
			learner[i].trainRunning[id] = true
			uniqueData[idx].IsTraining = true
		}
		learner[i].testIdxs = make([]int, len(setting.TrainingData))
		learner[i].testRunning = make(map[string]bool)
		for j, dataset := range setting.TestingData {
			id := dataset.ID()
			idx, ok := m[dataset.ID()]
			if !ok {
				uniqueIdx = addUniqueDataset(m, uniqueData, uniqueIdx, dataset)
				idx = uniqueIdx
			}
			learner[i].testIdxs[j] = idx
			learner[i].testRunning[id] = true
		}
	}
	return m, uniqueData, learner
}
*/
/*
func addUniqueDataset(m map[string]int, uniqueData []*datasetRunner, uniqueIdx int, newDataset Dataset) int {
	m[newDataset.ID()] = uniqueIdx
	uniqueIdx++
	runner := &datasetRunner{
		Set: newDataset,
		Idx: uniqueIdx,
	}
	uniqueData = append(uniqueData, runner)
	return uniqueIdx
}
*/

/*
func receiveData(learners []*learner, dataFinishedChan <-chan genComplete, mlChan chan<- mlRun, allFinished chan<- learner) {

	unfinishedIdx := make(map[int]struct{})
	for i := 0; i < len(learner); i++ {
		unfinishedIdx[i] = struct{}{}
	}
	launched := make([]bool, len(runners))
	nUnlaunched := len(runners)
	for done := range dataFinishedChan {
		// Log that this has been finished for all of the runners
		for idx := range unfinishedIdx {
			learner = learners[idx]
			learner.RegisterCompletion(done.ID())
			if learner.AllTrainDone() && launched[idx] == false {
				// All the training cases have finished, so launch the trainer
				mlChan <- mlRun{Idx: idx}
				launched[idx] = true
				nUnlaunched--
				if nUnlaunched == 0 {
					// All the runners launched, so close the channel to signal so
					close(mlChan)
				}
			}
			if learner.AllTrainDone() && learner.AllTestDone() {
				// Completely finished with this learner.
				delete(unfinishedIdx[idx])
				allFinished <- learner
			}
		}
	}
	if nUnlaunched != 0 {
		panic("logic wrong somewhere, not all of the learners lauched")
	}
	if len(unfinishedIdx) != 0 {
		panic("logic wrong somewhere, all learners not passed back")
	}
	return
}
*/

// MlTurb is the main script for learning and testing a ML model from data
func MlTurb(settings *Settings) error {
	// Do some error handling
	nTrainingSets := len(settings.TrainingData)
	if nTrainingSets == 0 {
		return errors.New("no training sets provided")
	}
	if len(settings.InputFeatures) == 0 {
		return errors.New("no input features")
	}
	if len(settings.OutputFeatures) == 0 {
		return errors.New("no output features")
	}

	// TODO: Add some fast path about checking for the existance of an algorithm
	// already (don't need to run training data, etc. if we already have a learned algorithm)

	// Generate the training data
	trainingDatasets := settings.TrainingData
	wgTrain := &sync.WaitGroup{}
	wgTrain.Add(1)
	var trainGenErr []error
	go func() {
		trainGenerators := GetGeneratables(trainingDatasets)
		trainGenErr = GenerateData(trainGenerators)
		trainGenErr = reduceError(trainGenErr)
		wgTrain.Done()
	}()
	// This should be removed later, but for now we have to do this because
	// we don't check to see if the same datasets are in testing and training
	wgTrain.Wait()

	// Generate the testing data
	testingDatasets := settings.TestingData

	wgTest := &sync.WaitGroup{}
	wgTest.Add(1)
	var testGenErr []error
	go func() {
		testGenerators := GetGeneratables(testingDatasets)
		testGenErr = GenerateData(testGenerators)
		testGenErr = reduceError(testGenErr)
		wgTest.Done()
	}()

	// Wait until all of the training runs are generated
	wgTrain.Wait()

	fmt.Println("Done generating training")

	if trainGenErr != nil {
		// Can't go on if there is an error generating the training data
		// Wait until the testing data is generated so computation time isn't wasted
		wgTest.Wait()
		return GenerationError{
			Training: trainGenErr,
			Testing:  testGenErr,
		}
	}

	// Load all of the training data
	inputs, outputs, weights, loadErrs := LoadTrainingData(trainingDatasets, DenseLoad, settings.InputFeatures,
		settings.OutputFeatures, settings.WeightFeatures, settings.WeightFunc)

	if loadErrs != nil {
		// Can't continue on if there were errors loading training data
		// Wait until the testing data is generated so computation time isn't wasted
		wgTest.Wait()
		return loadErrs
	}

	nSamples, inputDim := inputs.Dims()
	outputSamples, outputDim := outputs.Dims()
	weightSamples := len(weights)

	// Quick assertions. These are panics because they should never happen
	if outputSamples != nSamples {
		panic("sample size mismatch")
	}
	if weightSamples != nSamples && weightSamples != 0 {
		panic("weight size mismatch")
	}
	if inputDim != len(settings.InputFeatures) {
		panic("input dim size mismatch")
	}
	if outputDim != len(settings.OutputFeatures) {
		panic("output dim size mismatch")
	}

	fmt.Println("NumSamples: ", nSamples)
	fmt.Println("InputDim: ", inputDim)
	fmt.Println("OutputDim: ", outputDim)
	fmt.Println()

	algsavepath := PredictorDirectory(settings.Savepath)

	err := os.MkdirAll(algsavepath, 0700)
	if err != nil {
		wgTest.Wait()
		return err
	}

	algFile := PredictorFilename(settings.Savepath)

	// See if the training algorithm exists
	_, err = os.Open(algFile)
	if err != nil && !os.IsNotExist(err) {
		// There is an error and it's something other than the file doesn't exist
		wgTest.Wait()
		return errors.New("error trying to open trained algorithm: " + err.Error())
	} else {

	}
	if os.IsNotExist(err) {
		// The algorithm does not exist
		// TODO: Replace this by making algorithm a "trainer" so can more easily incorporate
		// FANN et al
		trainer := settings.Trainer
		var sp Predictor

		sp, err := trainer.Train(inputs, outputs, weights)
		if err != nil {
			wgTest.Wait()
			return errors.New("error training: " + err.Error())
		}
		fmt.Println("Before save")
		err = savePredictor(sp, algFile)
		if err != nil {
			wgTest.Wait()
			return errors.New("error saving predictor: " + err.Error())
		}
		fmt.Println("Done saving")
	} else {
		fmt.Println("Algorithm already trained")
	}

	// Load the training algorithm
	scalePredictor, err := LoadScalePredictor(settings.Savepath)
	if err != nil {
		wgTest.Wait()
		return errors.New("error loading training algorithm: " + err.Error())
	}

	// Wait for all of the training algorithms to finish computing
	wgTest.Wait()

	_ = scalePredictor

	// Do post processing and running new machine-learning algorithms
	// Post process the testing data
	postWg := &sync.WaitGroup{}
	postWg.Add(1)
	var postErr error
	go func() {
		fmt.Println("Starting post-process")
		postErr = postprocess(scalePredictor, settings)
		fmt.Println("Done post-process")
		postWg.Done()
	}()

	// Loop over all the testing sets, see if they are comparers, and
	// run them if they are

	compWg := &sync.WaitGroup{}
	compWg.Add(1)
	var compErr []error
	go func() {
		// Get the generators
		var generators []Generatable
		for i := range testingDatasets {
			outLoc := filepath.Join(settings.Savepath, "comparison")
			comp, ok := testingDatasets[i].(Comparable)
			if ok {

				gen, err := comp.Comparison(algFile, outLoc, settings.FeatureSet)
				if err != nil {
					panic(err)
				}
				generators = append(generators, gen)
			}
		}
		compErr = GenerateData(generators)
		for _, gen := range generators {
			p, ok := gen.(PostProcessor)
			if ok {
				fmt.Println("Ok posprocessor")
				err = p.PostProcess()
				if err != nil {
					panic(err)
				}
			}
		}
		compWg.Done()
	}()
	postWg.Wait()
	compWg.Wait()
	compErr = reduceError(compErr)
	if compErr != nil {
		return err
	}
	if postErr != nil {
		return err
	}

	// Should do something about post-processing

	return nil
}

// reduceError returns the slice of errors if any of the errors are non-nil
// and nil if all of the errors are nil
func reduceError(errs []error) []error {
	for i := range errs {
		if errs[i] != nil {
			return errs
		}
	}
	return nil
}

//
func savePredictor(sp Predictor, filename string) error {
	fmt.Println("save file name = ", filename)
	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	jsonBytes, err := json.MarshalIndent(sp, "", "\t")
	if err != nil {
		return err
	}
	f.Write(jsonBytes)
	f.Close()
	return nil
}

func PredictorDirectory(savepath string) string {
	return filepath.Join(savepath, "algorithm")
}

func PredictorFilename(savepath string) string {
	return filepath.Join(savepath, "algorithm", "trained_algorithm.json")
}

func LoadScalePredictor(savepath string) (ScalePredictor, error) {
	p := ScalePredictor{}
	filename := PredictorFilename(savepath)

	f, err := os.Open(filename)
	if err != nil {
		return p, err
	}
	defer f.Close()

	decoder := json.NewDecoder(f)

	decoder.Decode(&p)
	return p, nil
}
