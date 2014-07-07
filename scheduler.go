package ransuq

import (
	"fmt"
	"log"
	"runtime"
	"sync"
)

type Scheduler interface {
	//	Compute(Generatable)    // Blocks until job is started, but thread-safe
	//	Done() GenerateFinished // Blocks until job started, but thread-safe
	Launch() //
	Quit()
	AddChannel(GeneratableIO) // Adds a unique channel for generatable IO
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
