package plugins

import (
	"context"
	"fmt"
	"strings"

	logic "github.com/suryaumapathy2812/core-ai/agent/internal/plugins/logic"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type CurrentWeatherTool struct{}
type ForecastTool struct{}

func (t *CurrentWeatherTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "current_weather",
		Description: "Get current weather for a location.",
		StatusText:  "Checking weather...",
		Parameters:  []tools.Param{{Name: "location", Type: "string", Description: "Location name", Required: true}},
	}
}

func (t *CurrentWeatherTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	location, _ := call.Args["location"].(string)
	client := logic.WeatherClient{}
	current, err := client.Current(ctx, location)
	if err != nil {
		return tools.Fail("Weather lookup failed: "+err.Error(), nil)
	}
	out := fmt.Sprintf("%s: %s, %.1fC (feels like %.1fC), humidity %.0f%%, wind %.1f kph", current.Location, current.Condition, current.Temperature, current.FeelsLike, current.Humidity, current.WindKPH)
	return tools.Success(out, map[string]any{"location": current.Location, "temperature_c": current.Temperature})
}

func (t *ForecastTool) Definition() tools.Definition {
	return tools.Definition{
		Name:        "weather_forecast",
		Description: "Get weather forecast for a location for up to 7 days.",
		StatusText:  "Checking forecast...",
		Parameters: []tools.Param{
			{Name: "location", Type: "string", Description: "Location name", Required: true},
			{Name: "days", Type: "integer", Description: "Forecast days (1-7)", Required: false, Default: 3},
		},
	}
}

func (t *ForecastTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	location, _ := call.Args["location"].(string)
	days, _ := asInt(call.Args["days"])
	if days <= 0 {
		days = 3
	}
	client := logic.WeatherClient{}
	forecast, label, err := client.Forecast(ctx, location, days)
	if err != nil {
		return tools.Fail("Forecast lookup failed: "+err.Error(), nil)
	}
	lines := []string{fmt.Sprintf("Forecast for %s:", label)}
	for _, d := range forecast {
		lines = append(lines, fmt.Sprintf("- %s: %s, high %.1fC low %.1fC, rain %.1fmm, wind %.1fkph", d.Date, d.Condition, d.HighC, d.LowC, d.RainMM, d.WindKPH))
	}
	return tools.Success(strings.Join(lines, "\n"), map[string]any{"location": label, "days": len(forecast)})
}

var (
	_ tools.Tool = (*CurrentWeatherTool)(nil)
	_ tools.Tool = (*ForecastTool)(nil)
)
