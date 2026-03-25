package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
)

const (
	geocodeURL = "https://geocoding-api.open-meteo.com/v1/search"
	weatherURL = "https://api.open-meteo.com/v1/forecast"
)

var wmoCodes = map[int]string{
	0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Foggy",
	51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
	71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
	95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

type WeatherClient struct{ HTTP HTTPClient }

type CurrentWeather struct {
	Location    string
	Condition   string
	Temperature float64
	FeelsLike   float64
	Humidity    float64
	WindKPH     float64
}

type ForecastDay struct {
	Date      string
	Condition string
	HighC     float64
	LowC      float64
	RainMM    float64
	WindKPH   float64
}

func (w WeatherClient) Current(ctx context.Context, location string) (CurrentWeather, error) {
	geo, err := w.geocode(ctx, location)
	if err != nil {
		return CurrentWeather{}, err
	}

	q := url.Values{}
	q.Set("latitude", fmt.Sprintf("%f", geo.Lat))
	q.Set("longitude", fmt.Sprintf("%f", geo.Lon))
	q.Set("current", "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m")
	q.Set("timezone", geo.Timezone)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, weatherURL+"?"+q.Encode(), nil)
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return CurrentWeather{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return CurrentWeather{}, fmt.Errorf("weather api status: %d", resp.StatusCode)
	}

	var data struct {
		Current struct {
			Temp      float64 `json:"temperature_2m"`
			Humidity  float64 `json:"relative_humidity_2m"`
			FeelsLike float64 `json:"apparent_temperature"`
			Code      int     `json:"weather_code"`
			Wind      float64 `json:"wind_speed_10m"`
		} `json:"current"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return CurrentWeather{}, err
	}

	label := geo.Name
	if geo.Country != "" {
		label = geo.Name + ", " + geo.Country
	}

	return CurrentWeather{
		Location:    label,
		Condition:   wmoCodes[data.Current.Code],
		Temperature: data.Current.Temp,
		FeelsLike:   data.Current.FeelsLike,
		Humidity:    data.Current.Humidity,
		WindKPH:     data.Current.Wind,
	}, nil
}

func (w WeatherClient) Forecast(ctx context.Context, location string, days int) ([]ForecastDay, string, error) {
	if days < 1 {
		days = 1
	}
	if days > 7 {
		days = 7
	}
	geo, err := w.geocode(ctx, location)
	if err != nil {
		return nil, "", err
	}

	q := url.Values{}
	q.Set("latitude", fmt.Sprintf("%f", geo.Lat))
	q.Set("longitude", fmt.Sprintf("%f", geo.Lon))
	q.Set("daily", "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max")
	q.Set("timezone", geo.Timezone)
	q.Set("forecast_days", fmt.Sprintf("%d", days))
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, weatherURL+"?"+q.Encode(), nil)
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("weather api status: %d", resp.StatusCode)
	}

	var data struct {
		Daily struct {
			Dates []string  `json:"time"`
			Codes []int     `json:"weather_code"`
			Highs []float64 `json:"temperature_2m_max"`
			Lows  []float64 `json:"temperature_2m_min"`
			Rain  []float64 `json:"precipitation_sum"`
			Wind  []float64 `json:"wind_speed_10m_max"`
		} `json:"daily"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, "", err
	}

	out := make([]ForecastDay, 0, len(data.Daily.Dates))
	for i, date := range data.Daily.Dates {
		if i >= days {
			break
		}
		out = append(out, ForecastDay{
			Date:      date,
			Condition: wmoCodes[safeInt(data.Daily.Codes, i)],
			HighC:     safeFloat(data.Daily.Highs, i),
			LowC:      safeFloat(data.Daily.Lows, i),
			RainMM:    safeFloat(data.Daily.Rain, i),
			WindKPH:   safeFloat(data.Daily.Wind, i),
		})
	}
	label := geo.Name
	if geo.Country != "" {
		label += ", " + geo.Country
	}
	return out, label, nil
}

type geoResult struct {
	Name     string
	Country  string
	Timezone string
	Lat      float64
	Lon      float64
}

func (w WeatherClient) geocode(ctx context.Context, location string) (geoResult, error) {
	q := url.Values{}
	q.Set("name", location)
	q.Set("count", "1")
	q.Set("language", "en")
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, geocodeURL+"?"+q.Encode(), nil)
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return geoResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return geoResult{}, fmt.Errorf("geocode status: %d", resp.StatusCode)
	}
	var data struct {
		Results []struct {
			Name     string  `json:"name"`
			Country  string  `json:"country"`
			Timezone string  `json:"timezone"`
			Lat      float64 `json:"latitude"`
			Lon      float64 `json:"longitude"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return geoResult{}, err
	}
	if len(data.Results) == 0 {
		return geoResult{}, fmt.Errorf("location not found: %s", strings.TrimSpace(location))
	}
	r := data.Results[0]
	tz := r.Timezone
	if tz == "" {
		tz = "auto"
	}
	return geoResult{Name: r.Name, Country: r.Country, Timezone: tz, Lat: r.Lat, Lon: r.Lon}, nil
}

func safeFloat(s []float64, i int) float64 {
	if i < 0 || i >= len(s) {
		return 0
	}
	return s[i]
}

func safeInt(s []int, i int) int {
	if i < 0 || i >= len(s) {
		return 0
	}
	return s[i]
}
