package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
	"github.com/suryaumapathy2812/core-ai/agent/internal/observability"
)

type instrumentedDB struct {
	raw *sql.DB
}

func newInstrumentedDB(raw *sql.DB) *instrumentedDB {
	if raw == nil {
		return nil
	}
	return &instrumentedDB{raw: raw}
}

func (d *instrumentedDB) Close() error {
	if d == nil || d.raw == nil {
		return nil
	}
	return d.raw.Close()
}

func (d *instrumentedDB) ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	start := time.Now()
	res, err := d.raw.ExecContext(ctx, query, args...)
	logDB(ctx, "write", query, args, time.Since(start), err)
	return res, err
}

func (d *instrumentedDB) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	start := time.Now()
	rows, err := d.raw.QueryContext(ctx, query, args...)
	logDB(ctx, "read", query, args, time.Since(start), err)
	return rows, err
}

func (d *instrumentedDB) QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	start := time.Now()
	row := d.raw.QueryRowContext(ctx, query, args...)
	logDB(ctx, "read", query, args, time.Since(start), nil)
	return row
}

func (d *instrumentedDB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	start := time.Now()
	tx, err := d.raw.BeginTx(ctx, opts)
	logDB(ctx, "write", "BEGIN", nil, time.Since(start), err)
	return tx, err
}

func logDB(ctx context.Context, op string, query string, args []any, duration time.Duration, err error) {
	if !observability.ShouldLogDBQuery(query, err != nil) {
		return
	}
	e := log.Debug().Str("event", "db_query").Str("operation", op).Str("query", compactSQL(query)).Dur("duration", duration)
	if requestID := observability.RequestID(ctx); strings.TrimSpace(requestID) != "" {
		e.Str("request_id", requestID)
	}
	if len(args) > 0 {
		if b, mErr := json.Marshal(args); mErr == nil {
			e.RawJSON("args", b)
		}
	}
	if err != nil {
		e.Err(err).Msg("db query failed")
		return
	}
	e.Msg("db query")
}

func compactSQL(q string) string {
	q = strings.ReplaceAll(q, "\n", " ")
	q = strings.ReplaceAll(q, "\t", " ")
	for strings.Contains(q, "  ") {
		q = strings.ReplaceAll(q, "  ", " ")
	}
	return strings.TrimSpace(q)
}
