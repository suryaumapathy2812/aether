package tools

import (
	"context"
	"fmt"
	"strconv"
	"strings"
)

type Param struct {
	Name        string
	Type        string
	Description string
	Required    bool
	Default     any
	Enum        []string
	Items       map[string]any
}

type Definition struct {
	Name        string
	Description string
	StatusText  string
	Parameters  []Param
}

type Result struct {
	Output   string
	Metadata map[string]any
	Error    bool
}

// WithSandbox wraps the result's metadata with Arrow sandbox source code
// so the dashboard can render a rich UI for this tool output.
func (r Result) WithSandbox(source map[string]string) Result {
	if r.Metadata == nil {
		r.Metadata = map[string]any{}
	}
	r.Metadata["sandbox"] = map[string]any{
		"source":    source,
		"shadowDOM": true,
	}
	return r
}

func Success(output string, metadata map[string]any) Result {
	if metadata == nil {
		metadata = map[string]any{}
	}
	return Result{Output: output, Metadata: metadata, Error: false}
}

func Fail(msg string, metadata map[string]any) Result {
	if metadata == nil {
		metadata = map[string]any{}
	}
	return Result{Output: msg, Metadata: metadata, Error: true}
}

type Call struct {
	ID   string
	Args map[string]any
	Ctx  ExecContext
}

type Tool interface {
	Definition() Definition
	Execute(ctx context.Context, call Call) Result
}

func ValidateArgs(params []Param, args map[string]any) (map[string]any, error) {
	if args == nil {
		args = map[string]any{}
	}
	out := map[string]any{}
	for _, p := range params {
		v, ok := args[p.Name]
		if !ok {
			if p.Required {
				return nil, fmt.Errorf("missing required parameter: %s", p.Name)
			}
			if p.Default != nil {
				out[p.Name] = p.Default
			}
			continue
		}
		if err := validateType(p, v); err != nil {
			return nil, err
		}
		if len(p.Enum) > 0 {
			s, ok := v.(string)
			if !ok {
				return nil, fmt.Errorf("parameter %s must be string", p.Name)
			}
			if !contains(p.Enum, s) {
				return nil, fmt.Errorf("parameter %s must be one of: %s", p.Name, strings.Join(p.Enum, ", "))
			}
		}
		out[p.Name] = v
	}
	return out, nil
}

func validateType(p Param, v any) error {
	switch p.Type {
	case "string":
		if _, ok := v.(string); !ok {
			return fmt.Errorf("parameter %s must be string", p.Name)
		}
	case "integer":
		_, err := asInt(v)
		if err != nil {
			return fmt.Errorf("parameter %s must be integer", p.Name)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return fmt.Errorf("parameter %s must be boolean", p.Name)
		}
	case "array":
		switch v.(type) {
		case []any, []string:
		default:
			return fmt.Errorf("parameter %s must be array", p.Name)
		}
	case "object":
		if _, ok := v.(map[string]any); !ok {
			return fmt.Errorf("parameter %s must be object", p.Name)
		}
	}
	return nil
}

func contains(arr []string, v string) bool {
	for _, item := range arr {
		if item == v {
			return true
		}
	}
	return false
}

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
		return 0, fmt.Errorf("invalid integer")
	}
}
