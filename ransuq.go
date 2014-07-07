package ransuq

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
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

type GeneratableIO struct {
	In  chan Generatable
	Out chan GenerateFinished
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
		return
	}
	// It is a generatable, but it may be have already been generated. If so,
	// mark it as completed
	if generatable.Generated() {
		id := dataset.ID()
		log.Printf("%v has already been generated. Sending completion notice", id)
		d.norun <- dataset
		return
	}

	// Otherwise, send it to be generated
	d.compute <- generatable
}

// Done returns a dataset that is completed. Blocks until a dataset has been completed
func (d *DatasetRunner) Done() DatasetFinished {
	select {
	case data := <-d.norun:
		log.Printf("%v received without being generated", data.ID())
		return DatasetFinished{
			Dataset: data,
			Err:     nil,
		}
	case run := <-d.done:
		log.Printf("%v finished generating", run.Generatable.(Dataset).ID())
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

// MultiTurb runs a list of cases
// TODO: Needs to be some form of cluster calls
func MultiTurb(runs []*Settings, scheduler Scheduler) []error {
	scheduler.Launch()

	physicalDataCompute := make(chan Generatable)
	physicalDataDone := make(chan GenerateFinished)

	scheduler.AddChannel(GeneratableIO{In: physicalDataCompute, Out: physicalDataDone})

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

	mlChan := make(chan *mlRunData)
	allDoneChan := make(chan *learner)
	// Launch goroutine to read in training data as they come in and
	go func() {
		mlStarted := make([]bool, len(learners))

		stillRunning := make(map[int]struct{})
		for i := range learners {
			stillRunning[i] = struct{}{}
		}
		fmt.Println("Still Running = ", stillRunning)

		for i := 0; i < len(uniqueDatasets); i++ {
			run := datasetRunner.Done()
			fmt.Println("Still Running", stillRunning)
			for idx := range stillRunning {
				learners[idx].RegisterCompletion(run)

				if !mlStarted[idx] && learners[idx].AllTrainDone() {
					log.Printf("Settings case %v sent to training", idx)
					mlChan <- &mlRunData{Settings: learners[idx].Settings, trainErr: learners[idx].trainErrs, learningErr: nil}
					mlStarted[idx] = true
				}
				if learners[idx].AllTrainDone() && learners[idx].AllTestDone() {
					log.Printf("Settings case %v all data generated\n", idx)
					fmt.Printf("learner case %v: %v\n", idx, learners[idx])
					allDoneChan <- learners[idx]
					delete(stillRunning, idx)
				}
			}
		}
		if len(stillRunning) != 0 {
			panic("Cases still running after data generated")
		}
		close(mlChan)
		fmt.Println("Launching goroutine returned\n")
		// close(allDoneChan) // Don't need to close, will be garbage collected
		return
	}()

	//trainingCompleteChan := make(chan mlRunData)

	//doneTraining := make(chan bool)
	//doneGenTesting := make(chan bool)

	mlRunner := newMlRunner(scheduler)

	// Channel for returning ML results. This could be done better.
	mlDone := make(chan *mlRunData)

	//mlCtr := &mlRunningCounter{}

	// Read the ml runs that are coming in, and launch the training code
	go func() {
		for ml := range mlChan {
			// See if there are any training errors. If there are, skip the compute
			// process.
			var sent bool
			for _, err := range ml.trainErr {
				if err != nil {
					ml.learningErr = errors.New("error training")
					mlDone <- ml
					sent = true
					break
				}
			}
			if sent {
				continue
			}

			go func(ml *mlRunData) {
				mlRunner.Compute(ml)
				// Might not be the same one, but this at least guarantees that
				// the same number are sent and received.
				// TODO: This should be better with channel close. Need to improve
				// the scheduler
				// The way to do this is to have a "NewJobAllocator" or something
				// which returns a send only channel (for running jobs) and a
				// read-only channel for reading the results back in. Then, it's easy
				// to count the jobs right. The only trick is dealing with close, but
				// this is probably up to the user to ensure they aren't all closed before
				// they are sent and received ( like a normal single channel would)
			}(ml)

			// ... and this shows why we have a bad approach right now
			go func() {
				mlDone <- mlRunner.Done()
			}()
		}
		fmt.Println("ml run chan returned \n")
	}()

	// Channel for communicating everything is done
	finished := make(chan struct{})

	go func() {
		var sent int
		mlDoneMap := make(map[string]*mlRunData)
		testDoneMap := make(map[string]*learner)
		for {
			var newID string
			select {
			case mlRun := <-mlDone:
				newID = mlRun.ID()
				// Add the ID to the mlDoneMap
				mlDoneMap[newID] = mlRun

			case testDone := <-allDoneChan:
				// Add the ID to the testDoneMap
				fmt.Println("testDone = ", testDone)
				fakeMlRun := &mlRunData{Settings: testDone.Settings}
				newID = fakeMlRun.ID()
				testDoneMap[newID] = testDone
			}

			// See if the new returned case is in both maps, and if it is, launch
			// the post-processing step

			mlRun, mlOk := mlDoneMap[newID]
			testDone, testOk := testDoneMap[newID]

			if mlOk && testOk {
				sent++
				go func(mlRun *mlRunData, testDone *learner, finished chan struct{}) {
					runPostprocessing(scheduler, mlRun, testDone, finished)
				}(mlRun, testDone, finished)
				delete(mlDoneMap, newID)
				delete(testDoneMap, newID)
				if sent == len(runs) {
					fmt.Println("Postprocessing routine returned\n")
					return
				}
			}

		}
	}()

	// Read back that all of the postprocessing has finished
	for i := 0; i < len(runs); i++ {
		<-finished
		fmt.Println("Main routine read from finished. i = ", i)
	}

	// Lastly, collect the errors.
	errors := make([]error, len(runs))

	for i := 0; i < len(runs); i++ {
		errors[i] = learners[i].ReportError()
	}
	return errors
}

func runPostprocessing(scheduler Scheduler, mlRun *mlRunData, testDone *learner, finished chan struct{}) {
	// To do post processing, we need to:
	// 		launch all of the comparison jobs (if any)
	//		Run all of the post-processing data
	// These can be done concurrently. Use a waitgroup to synchronize finishing

	// First, check if there was an error, if so, can't do any of the post-processing
	if mlRun.learningErr != nil {
		testDone.learningErr = mlRun.learningErr
		finished <- struct{}{}
		return
	}

	wg := sync.WaitGroup{}

	// Launch all of the comparison jobs
	wg.Add(1)
	go func() {
		// Create read and reply channels and add them to the scheduler
		g := GeneratableIO{make(chan Generatable), make(chan GenerateFinished)}
		scheduler.AddChannel(g)

		var skipped int
		for _, test := range mlRun.Settings.TestingData {
			// See if the test data is comparable
			comp, ok := test.(Comparable)

			if !ok {
				skipped++
				continue
			}
			// If it is, set it up and send it
			outLoc := filepath.Join(testDone.Settings.Savepath, "comparison")
			algFile := PredictorFilename(testDone.Settings.Savepath)

			gen, err := comp.Comparison(algFile, outLoc, testDone.Settings.FeatureSet)
			if err != nil {
				go func() { g.Out <- GenerateFinished{gen, err} }()
				continue
			}

			// See if it's aleady been run
			if gen.Generated() {
				go func() { g.Out <- GenerateFinished{gen, nil} }()
				continue
			}
			log.Println("Sending case to comparison: ", gen.ID())
			go func() { g.In <- gen }()
		}

		postprocessWg := &sync.WaitGroup{}
		for i := skipped; i < len(mlRun.Settings.TestingData); i++ {
			// TODO: Fix this so that the order isn't messed up
			gf := <-g.Out
			log.Println("Case read from comparison: ", gf.ID())
			if gf.Err != nil {
				testDone.comparisonErrs[i] = gf.Err
				continue
			}
			postprocessWg.Add(1)
			go func(i int) {
				p, ok := gf.Generatable.(PostProcessor)
				if ok {
					fmt.Println("Launching postprocess: " + gf.ID())
					err := p.PostProcess()
					fmt.Println("Done postprocessing")
					if err != nil {
						testDone.comparisonErrs[i] = err
					}
				}
				postprocessWg.Done()
			}(i)
		}
		fmt.Println("Waiting for postprocess to be done")
		postprocessWg.Wait()
		fmt.Println("Postprocess is done")
		wg.Done()
		fmt.Println("Past call to done 1")
	}()

	// Second thing to do is to run the normal post-processing stuff
	wg.Add(1)
	go func() {
		defer wg.Done()
		fmt.Println("Starting training data post processing")
		scalePredictor, err := LoadScalePredictor(testDone.Settings.Savepath)
		if err != nil {
			testDone.postprocessErr = err
			runtime.GC()
			return
		}
		postErr := postprocess(scalePredictor, testDone.Settings)
		testDone.postprocessErr = postErr
	}()
	runtime.GC()
	fmt.Println("Main postprocess routine reached call to wait")
	wg.Wait()
	fmt.Println("Main postprocess routine finished call to wait")
	finished <- struct{}{}
	fmt.Println("postprocess routine sent signal to finished")
}

type mlRunData struct {
	*Settings
	trainErr    []error
	learningErr error
}

func (m *mlRunData) Generated() bool {
	// Checks if the machine learning has already been run
	algFile := PredictorFilename(m.Settings.Savepath)

	_, err := os.Open(algFile)
	if err != nil {
		fmt.Println("Algfile is: ", algFile, " err is ", err.Error())
	}
	return err == nil // If we can open the alg file it's because it has been generated
}

func (m *mlRunData) ID() string {
	// Say that the ID is the same as the predictor filename, should be unique
	return PredictorFilename(m.Settings.Savepath)
}

func (m *mlRunData) NumCores() int {
	return runtime.GOMAXPROCS(0) - 1 // -1 so that we can also start postprocess routines at the same time
}

func (m *mlRunData) Run() error {

	algsavepath := PredictorDirectory(m.Settings.Savepath)
	algFile := PredictorFilename(m.Settings.Savepath)
	err := os.MkdirAll(algsavepath, 0700)
	if err != nil {
		return err
	}

	settings := m.Settings

	fmt.Println("Settings.TrainingData is:")
	for _, dat := range settings.TrainingData {
		fmt.Println(dat.ID())
	}
	// Load all of the training data
	inputs, outputs, weights, loadErrs := LoadTrainingData(settings.TrainingData, DenseLoad,
		settings.InputFeatures, settings.OutputFeatures, settings.WeightFeatures, settings.WeightFunc)

	if loadErrs != nil {
		return loadErrs
	}

	nRow, nCol := inputs.Dims()
	fmt.Println("Calling train with ", nRow, " rows and ", nCol, " columns")

	trainer := settings.Trainer
	sp, result, err := trainer.Train(inputs, outputs, weights)
	if err != nil {
		return err
	}

	resultFile := filepath.Join(algsavepath, "train_result.json")
	err = savePredictor(sp, result, algFile, resultFile)
	if err != nil {
		return errors.New("error saving predictor: " + err.Error())
	}

	// Make a plot of pred vs. truth and err. vs. truth over the training data
	path := filepath.Join(m.Settings.Savepath, "postprocess", "trainingData")
	ptrSP := sp.(*ScalePredictor)
	// TODO: Fix this. It is really ugly.
	makeComparisons(inputs, outputs, *ptrSP, settings.InputFeatures, settings.OutputFeatures, path)
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

type mlRunnerStruct struct {
	computeChan chan Generatable
	doneChan    chan GenerateFinished
}

// Generates a new mlrunner and adds it to the scheduler
func newMlRunner(scheduler Scheduler) *mlRunnerStruct {
	cc := make(chan Generatable)
	dc := make(chan GenerateFinished)

	scheduler.AddChannel(GeneratableIO{In: cc, Out: dc})
	return &mlRunnerStruct{
		computeChan: cc,
		doneChan:    dc,
	}
}
func (m *mlRunnerStruct) Close() {
	close(m.computeChan)
}

func (m *mlRunnerStruct) Compute(ml *mlRunData) {
	log.Print("ml " + ml.ID() + "received for generating")
	errStr := ""
	for i, err := range ml.trainErr {
		if err != nil {
			if errStr == "" {
				errStr = "Error training"
			}
			errStr += " case " + strconv.Itoa(i) + "err: " + err.Error()
		}
	}
	if errStr != "" {
		log.Print("ml " + ml.ID() + "had training generation error")
		// There was an error training, so send it as done
		ml.learningErr = errors.New(errStr)
		m.doneChan <- GenerateFinished{Generatable: ml, Err: nil}
		return
	}
	if ml.Generated() {
		log.Print("ml " + ml.ID() + "has already been generated")
		m.doneChan <- GenerateFinished{Generatable: ml, Err: nil}
		return
	}
	log.Print("ml " + ml.ID() + "sent to compute")
	m.computeChan <- ml
}

func (m *mlRunnerStruct) Done() *mlRunData {
	data := <-m.doneChan
	err := data.Err

	ml := data.Generatable.(*mlRunData)
	ml.learningErr = err
	return ml
}

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

	learningErr    error
	comparisonErrs []error
	postprocessErr error
}

func (l *learner) ReportError() error {
	var str string
	var isTrainingError bool
	for _, err := range l.trainErrs {
		if err != nil {
			if isTrainingError == false {
				str += "error training: "
			}
			str += err.Error()
			isTrainingError = true
		}
	}
	if isTrainingError {
		str += " "
	}
	var isTestingError bool
	for _, err := range l.testErrs {
		if err != nil {
			if isTestingError {
				str += "error testing: "
			}
			str += err.Error()
			isTestingError = true
		}
	}
	if isTrainingError || isTestingError {
		return errors.New(str)
	}

	if l.learningErr != nil {
		return l.learningErr
	}

	isPostprocessError := l.postprocessErr != nil
	if isPostprocessError {
		str += "Postprocessing error: " + l.postprocessErr.Error() + " "
	}

	var isCompError bool
	for _, err := range l.comparisonErrs {
		if err != nil {
			if isCompError == false {
				str += "error comparing: "
			}
			str += err.Error()
			isCompError = true
		}
	}

	if isPostprocessError || isCompError {
		return errors.New(str)
	}

	return nil
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
	for i := range learners {
		learners[i] = &learner{}
	}
	var uniqueIdx int
	for i, setting := range runs {
		learners[i].Settings = setting
		learners[i].trainRunning = make(map[string]int)
		learners[i].trainIdx = make([]int, len(setting.TrainingData))
		learners[i].trainErrs = make([]error, len(setting.TrainingData))
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
		learners[i].testRunning = make(map[string]int)
		learners[i].testIdx = make([]int, len(setting.TestingData))
		learners[i].testErrs = make([]error, len(setting.TestingData))
		learners[i].comparisonErrs = make([]error, len(setting.TestingData))
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

//
func savePredictor(sp Predictor, result TrainResults, filename, resultfilename string) error {
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

	f2, err := os.Create(resultfilename)
	if err != nil {
		return err
	}

	jsonBytes, err = json.MarshalIndent(result, "", "\t")
	if err != nil {
		return err
	}
	_, err = f2.Write(jsonBytes)
	if err != nil {
		panic(err)
	}
	f2.Close()

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
	err = decoder.Decode(&p)
	if err != nil {
		return p, err
	}
	return p, nil
}
