package plugins

import (
	"context"
	"fmt"

	logic "github.com/suryaumapathy2812/core-ai/agent/internal/plugins/logic"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

type WolframQueryTool struct{}
type CurrencyConvertTool struct{}

func (t *WolframQueryTool) Definition() tools.Definition {
	return tools.Definition{Name: "wolfram_query", Description: "Query Wolfram Alpha for computational answers.", StatusText: "Computing answer...", Parameters: []tools.Param{{Name: "query", Type: "string", Description: "Computation query", Required: true}}}
}

func (t *WolframQueryTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	query, _ := call.Args["query"].(string)
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	appID, err := maybeDecryptFromState(ctx, call, cfg["wolfram_app_id"])
	if err != nil {
		return tools.Fail("Wolfram is not configured: invalid wolfram_app_id", nil)
	}
	if appID == "" {
		return tools.Fail("Wolfram is not configured: missing wolfram_app_id", nil)
	}
	exchangeKey, _ := maybeDecryptFromState(ctx, call, cfg["exchangerate_api_key"])
	client := logic.WolframClient{WolframAppID: appID, ExchangeRateKey: exchangeKey}
	ans, err := client.Query(ctx, query)
	if err != nil {
		return tools.Fail("Wolfram query failed: "+err.Error(), nil)
	}
	return tools.Success(fmt.Sprintf("Wolfram: %s\n\nAnswer: %s", query, ans), map[string]any{"query": query, "answer": ans})
}

func (t *CurrencyConvertTool) Definition() tools.Definition {
	return tools.Definition{Name: "currency_convert", Description: "Convert between currencies using live rates.", StatusText: "Converting currency...", Parameters: []tools.Param{{Name: "amount", Type: "integer", Description: "Amount to convert", Required: true}, {Name: "from_currency", Type: "string", Description: "Source currency code", Required: true}, {Name: "to_currency", Type: "string", Description: "Target currency code", Required: true}}}
}

func (t *CurrencyConvertTool) Execute(ctx context.Context, call tools.Call) tools.Result {
	amountInt, _ := asInt(call.Args["amount"])
	from, _ := call.Args["from_currency"].(string)
	to, _ := call.Args["to_currency"].(string)
	cfg, err := pluginConfig(ctx, call)
	if err != nil {
		return tools.Fail(err.Error(), nil)
	}
	appID, _ := maybeDecryptFromState(ctx, call, cfg["wolfram_app_id"])
	exchangeKey, _ := maybeDecryptFromState(ctx, call, cfg["exchangerate_api_key"])
	client := logic.WolframClient{WolframAppID: appID, ExchangeRateKey: exchangeKey}
	res, err := client.CurrencyConvert(ctx, float64(amountInt), from, to)
	if err != nil {
		return tools.Fail("Currency conversion failed: "+err.Error(), nil)
	}
	out := fmt.Sprintf("%.2f %s = %.2f %s (rate %.6f, source %s)", res.Amount, res.FromCurrency, res.Converted, res.ToCurrency, res.Rate, res.Source)
	return tools.Success(out, map[string]any{"amount": res.Amount, "from_currency": res.FromCurrency, "to_currency": res.ToCurrency, "converted": res.Converted, "rate": res.Rate})
}

var (
	_ tools.Tool = (*WolframQueryTool)(nil)
	_ tools.Tool = (*CurrencyConvertTool)(nil)
)
