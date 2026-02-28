package builtin

import (
	"context"
	"fmt"
	"time"

	"github.com/suryaumapathy/core-ai/agent/internal/tools"
)

type WorldTimeTool struct{}

func (t *WorldTimeTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "world_time",
		Description: "Get current date and time in a timezone.",
		StatusText:  "Checking time...",
		Parameters: []tools.Param{
			{Name: "timezone", Type: "string", Description: "IANA timezone (e.g. Asia/Kolkata)", Required: false, Default: "UTC"},
		},
	}
}

func (t *WorldTimeTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	_ = ctx
	tz, _ := call.Args["timezone"].(string)
	if tz == "" {
		tz = "UTC"
	}
	loc, err := time.LoadLocation(tz)
	if err != nil {
		return tools.Fail("Invalid timezone: "+tz, nil)
	}
	now := time.Now().In(loc)
	out := fmt.Sprintf("Current time in %s is %s", tz, now.Format(time.RFC1123Z))
	return tools.Success(out, map[string]any{"timezone": tz, "iso": now.Format(time.RFC3339)})
}

var _ tools.Tool = (*WorldTimeTool)(nil)
