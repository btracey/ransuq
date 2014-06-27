package settings

import (
	"sort"
)

func init() {
	sortedWeights = append(sortedWeights, NoWeight)

	sort.Strings(sortedWeights)
}

const (
	NoWeight = "none"
)

var sortedWeights []string

// Weight is the weight setting. WeightFeatures is a list of the features that
// the weight dependns on. weightFunc is a mapping from the values of those features
// to an output weight
func GetWeight(weight string) (weightFeatures []string, weightFunc func([]float64) float64, err error) {
	switch weight {
	default:
		return nil, nil, Missing{
			Prefix:  "weight setting not found",
			Options: sortedWeights,
		}
	case "none":
		return nil, nil, nil
	}
}
