package settings

const (
	NoExtraStrings        = "none"
	FlatplateBlOnlyCutoff = "FlatplateBlOnlyCutoff"
)

func GetSU2ExtraStrings(strtype string) ([]string, error) {
	switch strtype {
	default:
		return nil, Missing{
			Prefix: "extrastrings setting not found",
		}
	case NoExtraStrings:
		return []string{}, nil
	case FlatplateBlOnlyCutoff:
		return []string{"FlatplateBlOnlyCutoff"}, nil
	}
}
