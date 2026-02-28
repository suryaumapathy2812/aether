package plugins

import (
	"fmt"
	"strconv"
	"strings"
)

func asInt(v any) (int, error) {
	switch n := v.(type) {
	case int:
		return n, nil
	case int64:
		return int(n), nil
	case float64:
		return int(n), nil
	case string:
		i, err := strconv.Atoi(strings.TrimSpace(n))
		if err != nil {
			return 0, err
		}
		return i, nil
	default:
		return 0, fmt.Errorf("invalid int")
	}
}

func asString(v any) (string, error) {
	s, ok := v.(string)
	if !ok {
		return "", fmt.Errorf("invalid string")
	}
	return s, nil
}

func asBool(v any) (bool, error) {
	b, ok := v.(bool)
	if ok {
		return b, nil
	}
	s, ok := v.(string)
	if !ok {
		return false, fmt.Errorf("invalid bool")
	}
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "true", "1", "yes", "y":
		return true, nil
	case "false", "0", "no", "n":
		return false, nil
	default:
		return false, fmt.Errorf("invalid bool")
	}
}
