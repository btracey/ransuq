package ransuq

import "fmt"

type ErrorList []error

func (e ErrorList) Error() string {
	var str string
	for i, err := range e {
		if err != nil {
			str += fmt.Sprintf("  case %d: %s", i, err.Error())
		}
	}
	return str
}

func (e ErrorList) AllNil() bool {
	if len(e) == 0 {
		return true
	}
	for _, err := range e {
		if err != nil {
			return false
		}
	}
	return true
}

type PostprocessError struct {
	Testing  ErrorList
	Training ErrorList
}

func (p PostprocessError) Error() string {
	var str string
	if p.Training != nil && p.Testing != nil {
		str += "error postprocessing"
	}
	if p.Training != nil {
		str += "training: " + p.Training.Error()
	}
	if p.Testing != nil {
		str += "testing: " + p.Testing.Error()
	}
	return str
}
