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

type Scheduler interface {
	//	Compute(Generatable)    // Blocks until job is started, but thread-safe
	//	Done() GenerateFinished // Blocks until job started, but thread-safe
	Launch() //
	Quit()
	AddChannel(GeneratableIO) // Adds a unique channel for generatable IO
}

type GeneratableIO struct {
	In  chan Generatable
	Out chan GenerateFinished
}

// Local Scheduler is a scheduler that assumes a shared-memory environment for
//
type LocalScheduler struct {
	//compute         chan Generatable
	//done            chan GenerateFinished
	nCores          int
	nAvailableCores int
	cond            *sync.Cond
	mux             *sync.Mutex

	addMux *sync.RWMutex

	quit chan struct{}
	gen  chan generateChanIdx
	//done chan generateChanIdx

	genIOs []GeneratableIO
	//quitGen []chan struct{}
	wgs []*sync.WaitGroup
}

type generateChanIdx struct {
	Gen Generatable
	Idx int
	Err error
}

func NewLocalScheduler() *LocalScheduler {
	l := &LocalScheduler{
		nCores:          runtime.GOMAXPROCS(0),
		nAvailableCores: runtime.GOMAXPROCS(0),
		mux:             &sync.Mutex{},
		quit:            make(chan struct{}),
		gen:             make(chan generateChanIdx),
		//done:            make(chan GenerateFinished),
		addMux: &sync.RWMutex{},
	}
	l.cond = sync.NewCond(l.mux)
	return l
}

func (l *LocalScheduler) Launch() {
	go func() {
		l.compute()
	}()
}

func (l *LocalScheduler) Quit() {
	l.quit <- struct{}{}
}

/*
func (l *LocalScheduler) Compute(gen Generatable) {
	l.gen <- gen
}
*/

// Adds a channel where generatables can be sent and retrived on a specific out
// stream
func (l *LocalScheduler) AddChannel(g GeneratableIO) {
	l.addMux.Lock()
	idx := len(l.genIOs)
	l.genIOs = append(l.genIOs, g)
	l.wgs = append(l.wgs, &sync.WaitGroup{})
	l.addMux.Unlock()

	// Launch the data type for sending new compute tasks
	go func(idx int) {
		l.addMux.RLock()
		c := l.genIOs[idx].In
		wg := l.wgs[idx]
		l.addMux.RUnlock()

		for gen := range c {
			wg.Add(1)
			// Send the task to be computed
			l.gen <- generateChanIdx{gen, idx, nil}
		}
		// The receive channel has been closed. Wait until all of the tasks are done
		// and then close the read chan
		l.addMux.RLock()
		l.wgs[idx].Wait()
		close(l.genIOs[idx].Out)
		l.addMux.RUnlock()
	}(idx)
}

/*
func (l *LocalScheduler) Done() GenerateFinished {
	return <-l.done
}
*/

func (l *LocalScheduler) compute() {
	for {
		select {
		case <-l.quit:
			return
		case gen := <-l.gen:
			neededCores := gen.Gen.NumCores()
			if neededCores > l.nCores {
				str := fmt.Sprintf("not enough available cores: %v requested %v available. generatable: %v", neededCores, l.nCores, gen.Gen.ID())
				panic(str)
			}
			log.Print("gen received")
			fmt.Println("ncores = ", l.nCores)
			fmt.Println("avail cores = ", l.nAvailableCores)
			// Wait until a processor is available
			// TODO: Need to change this such that it doesn't block, but lets
			// smaller jobs through.
			l.cond.L.Lock()
			for l.nAvailableCores < neededCores {
				l.cond.Wait()
			}
			l.nAvailableCores -= neededCores
			if l.nAvailableCores < 0 {
				panic("nAvail should never be negative")
			}
			// Launch the case
			go func(genIdx generateChanIdx) {
				gen := genIdx.Gen
				// Run the case
				log.Print("Scheduler started " + gen.ID())
				err := gen.Run()
				log.Print("Scheduler finished " + gen.ID())
				// Tell the scheduler that the cores are free again, and signal
				// to the waiting goroutines that they can wait
				l.mux.Lock()
				l.nAvailableCores += gen.NumCores()
				fmt.Println(gen.NumCores(), " readded to available")
				l.mux.Unlock()
				l.cond.Broadcast()

				// Send the finished case back on the proper channel. Make sure
				// we aren't appending at the same time
				l.addMux.RLock()
				l.wgs[genIdx.Idx].Done()
				l.genIOs[genIdx.Idx].Out <- GenerateFinished{gen, err}
				l.addMux.RUnlock()
			}(gen)
			l.cond.L.Unlock()
		}
	}
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
		fmt.Println("Starting training data post processing")
		scalePredictor, err := LoadScalePredictor(testDone.Settings.Savepath)
		//if err != nil {
		if true {
			testDone.postprocessErr = err
			runtime.GC()
			return
		}
		postErr := postprocess(scalePredictor, testDone.Settings)
		testDone.postprocessErr = postErr
		fmt.Println("Done data postprocessing")
		wg.Done()
		fmt.Println("Past call to done 2")
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
	return 6 // Need to do more
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

		sp, result, err := trainer.Train(inputs, outputs, weights)
		if err != nil {
			wgTest.Wait()
			return errors.New("error training: " + err.Error())
		}
		resultFile := filepath.Join(algsavepath, "train_result.json")
		err = savePredictor(sp, result, algFile, resultFile)
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
