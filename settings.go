package ransuq

// ConvergenceSettings controls how finely the machine learning should be run
type TrainSettings struct {
	ObjAbsTol   float64
	GradAbsTol  float64
	MaxFunEvals int // Maximum function evaluations
}

// TODO: Need to think about all of this more. Where is the line between the different
// datatypes

type Settings struct {
	FeatureSet     string // Identifier to pass to comparison
	TrainingData   []Dataset
	TestingData    []Dataset
	InputFeatures  []string
	OutputFeatures []string
	WeightFeatures []string
	WeightFunc     func([]float64) float64
	Savepath       string // Location of where to save the algorithm and plots

	Trainer *Trainer
}
