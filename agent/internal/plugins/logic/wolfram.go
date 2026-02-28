package logic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

const (
	wolframAPI      = "https://api.wolframalpha.com/v1/result"
	exchangeRateAPI = "https://v6.exchangerate-api.com/v6"
)

type WolframClient struct {
	HTTP            HTTPClient
	WolframAppID    string
	ExchangeRateKey string
}

type CurrencyResult struct {
	Amount       float64
	FromCurrency string
	ToCurrency   string
	Converted    float64
	Rate         float64
	Source       string
}

func (w WolframClient) Query(ctx context.Context, query string) (string, error) {
	if strings.TrimSpace(w.WolframAppID) == "" {
		return "", fmt.Errorf("missing wolfram app id")
	}
	v := url.Values{}
	v.Set("input", query)
	v.Set("appid", w.WolframAppID)
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, wolframAPI+"?"+v.Encode(), nil)
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotImplemented {
		return "", fmt.Errorf("wolfram could not compute: %s", query)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("wolfram status: %d", resp.StatusCode)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	ans := strings.TrimSpace(string(b))
	if ans == "" {
		return "", fmt.Errorf("wolfram returned empty result")
	}
	return ans, nil
}

func (w WolframClient) CurrencyConvert(ctx context.Context, amount float64, fromCurrency, toCurrency string) (CurrencyResult, error) {
	fromCurrency = strings.ToUpper(strings.TrimSpace(fromCurrency))
	toCurrency = strings.ToUpper(strings.TrimSpace(toCurrency))

	if strings.TrimSpace(w.ExchangeRateKey) != "" {
		url := fmt.Sprintf("%s/%s/pair/%s/%s/%f", exchangeRateAPI, w.ExchangeRateKey, fromCurrency, toCurrency, amount)
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		resp, err := defaultHTTPClient(w.HTTP).Do(req)
		if err != nil {
			return CurrencyResult{}, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return CurrencyResult{}, fmt.Errorf("exchange api status: %d", resp.StatusCode)
		}
		var payload struct {
			Result           string  `json:"result"`
			ConversionRate   float64 `json:"conversion_rate"`
			ConversionResult float64 `json:"conversion_result"`
			ErrorType        string  `json:"error-type"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
			return CurrencyResult{}, err
		}
		if payload.Result != "success" {
			return CurrencyResult{}, fmt.Errorf("currency conversion failed: %s", payload.ErrorType)
		}
		return CurrencyResult{Amount: amount, FromCurrency: fromCurrency, ToCurrency: toCurrency, Converted: payload.ConversionResult, Rate: payload.ConversionRate, Source: "exchangerate-api"}, nil
	}

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, "https://open.er-api.com/v6/latest/"+fromCurrency, nil)
	resp, err := defaultHTTPClient(w.HTTP).Do(req)
	if err != nil {
		return CurrencyResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return CurrencyResult{}, fmt.Errorf("open er api status: %d", resp.StatusCode)
	}
	var payload struct {
		Rates map[string]float64 `json:"rates"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return CurrencyResult{}, err
	}
	rate, ok := payload.Rates[toCurrency]
	if !ok {
		return CurrencyResult{}, fmt.Errorf("currency not found: %s", toCurrency)
	}
	return CurrencyResult{Amount: amount, FromCurrency: fromCurrency, ToCurrency: toCurrency, Converted: amount * rate, Rate: rate, Source: "open-er-api"}, nil
}
