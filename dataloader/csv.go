package dataloader

import (
	"bufio"
	"errors"
	"os"
	"strconv"
	"strings"
)

/*

// CSV is a wrapper around numcsv. It will read in the data etc. when it is allocated
// to allow good wrapping with dataloader
type NumCSV struct {
	IgnoreFunc func(d []float64) bool

	headings []string
	data     *mat64.Dense
	filename string
}

func NewCSV(n *numcsv.Reader, filename string) (*NumCSV, error) {
	headings, err := n.ReadHeading()
	if err != nil {
		return nil, err
	}
	data, err := n.ReadAll()
	if err != nil {
		return nil, err
	}

	return &NumCSV{
		headings: headings,
		data:     data,
		filename: filename,
	}, nil
}

// Fieldmap just returns the string. For a NumCSV, all headers must be there
// explicitly
func (n *NumCSV) Fieldmap(str string) *FieldTransformer {
	return &FieldTransformer{
		InternalNames: []string{str},
		Transformer:   identityFunc,
	}
}

func (n *NumCSV) ReadFields(fields []string, filename string) ([][]float64, error) {
	if filename != n.filename {
		return nil, errors.New("Filename does not match the initialized filename")
	}
	r, _ := n.data.Dims()

	data := make([][]float64, r)
	for i := range data {
		data[i] = make([]float64, len(fields))
	}

	for j, str := range fields {
		idx := findStrIdx(n.headings, str)
		if idx == -1 {
			return nil, errors.New("Fieldname " + str + " not found")
		}
		for i := 0; i < r; i++ {
			data[i][j] = n.data.At(i, idx)
		}
	}
	return data, nil
}

func findStrIdx(strs []string, str string) int {
	for i, s := range strs {
		if s == str {
			return i
		}
	}
	return -1
}
*/

// CSV is a simple type for loading CSV files. It assumes the data are number-
// valued, and so it is looser on other formating (not as strict on
// whitespace, etc.)
type NaiveCSV struct {
	// If there is an additional separator beyond whitespace (default to ,).
	// If the data are whitespace separated this has no effect
	Delimiter string
	FieldMap  map[string]string
}

// Fieldmap just returns the string. For a naive CSV, all headers must be there
// explicitly
func (c *NaiveCSV) Fieldmap(str string) *FieldTransformer {

	var internalName string
	if c.FieldMap == nil {
		internalName = str
	} else {
		var ok bool
		internalName, ok = c.FieldMap[str]
		if !ok {
			internalName = str
		}
	}
	return &FieldTransformer{
		InternalNames: []string{internalName},
		Transformer:   identityFunc,
	}
}

func (c *NaiveCSV) ReadFields(fields []string, filename string) ([][]float64, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(f)

	notdone := scanner.Scan()
	if !notdone {
		return nil, errors.New("No fields line")
	}
	if scanner.Err() != nil {
		return nil, errors.New("error parsing field line")
	}

	text := scanner.Text()
	strs := c.splitLine(text, true)

	// The elements to strs are the fieldnames. Make a map from the string to
	// which column it is
	nFileFields := len(strs)
	fieldMap := make(map[string]int)
	for i := range strs {
		_, ok := fieldMap[strs[i]]
		if ok {
			return nil, errors.New("duplicate field: " + strs[i])
		}
		fieldMap[strs[i]] = i
	}

	// Now that we have all the field names, check that all of the needed fields
	// are in the map
	nNeededFields := len(fields)
	idxs := make([]int, nNeededFields)
	for i, field := range fields {
		var ok bool
		idxs[i], ok = fieldMap[field]
		if !ok {
			return nil, errors.New("field: " + field + " not in map")
		}
	}

	data := make([][]float64, 0)

	count := 0
	for scanner.Scan() {
		text := scanner.Text()
		strs := c.splitLine(text, false)

		count++

		if len(strs) != nFileFields {
			str := "incorrect number of numbers. number of string fields is: " + strconv.Itoa(nFileFields) + " number of numbers is " + strconv.Itoa(len(strs))
			return nil, errors.New(str)
		}

		// Now, extract the data
		newData := make([]float64, nNeededFields)
		for i, idx := range idxs {
			var err error
			newData[i], err = strconv.ParseFloat(strs[idx], 64)
			if err != nil {
				return nil, errors.New("formatting error: string is " + strs[idx])
			}
		}

		data = append(data, newData)
	}
	if scanner.Err() != nil {
		return nil, errors.New("scanning data: " + err.Error())
	}
	return data, nil
}

func (c *NaiveCSV) splitLine(text string, trimQuote bool) []string {
	delimiter := c.Delimiter
	if delimiter == "" {
		delimiter = ","
	}

	// Split first at delimiter
	strs := strings.Split(text, delimiter)

	if len(strs) == 1 {
		// Try splitting on whitespace
		strs = strings.Fields(text)
	}

	// Trim white space around the values, and the quotations around the fields
	// if they exis
	text = strings.TrimSpace(text)
	for i, s := range strs {
		newStr := strings.TrimSpace(s)
		if trimQuote {
			newStr = strings.TrimPrefix(newStr, "\"")
			newStr = strings.TrimSuffix(newStr, "\"")
		}
		strs[i] = newStr
	}
	return strs
}
