package auth

import (
	"errors"
	"net/http"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Identity struct {
	UserID   string
	DeviceID string
	Token    string
}

type Authenticator struct {
	db *pgxpool.Pool
}

func New(db *pgxpool.Pool) *Authenticator {
	return &Authenticator{db: db}
}

func (a *Authenticator) IdentityFromRequest(r *http.Request) (Identity, error) {
	tokens := ExtractTokens(r)
	if len(tokens) == 0 {
		return Identity{}, errors.New("Missing authorization")
	}

	for _, token := range tokens {
		if token == "" {
			continue
		}
		var userID string
		err := a.db.QueryRow(r.Context(), `
			SELECT user_id
			FROM session
			WHERE token = $1 AND expires_at > now()
			LIMIT 1
		`, token).Scan(&userID)
		if err == nil && strings.TrimSpace(userID) != "" {
			return Identity{UserID: userID, Token: token}, nil
		}

		var deviceID string
		err = a.db.QueryRow(r.Context(), `
			SELECT id, user_id
			FROM devices
			WHERE token = $1
			LIMIT 1
		`, token).Scan(&deviceID, &userID)
		if err == nil && strings.TrimSpace(userID) != "" {
			_, _ = a.db.Exec(r.Context(), "UPDATE devices SET last_seen = now() WHERE token = $1", token)
			return Identity{UserID: userID, DeviceID: deviceID, Token: token}, nil
		}
		if err != nil && !errors.Is(err, pgx.ErrNoRows) {
			return Identity{}, err
		}
	}

	return Identity{}, errors.New("Invalid or expired session")
}

func ExtractTokens(r *http.Request) []string {
	out := make([]string, 0, 3)
	if t := BearerToken(r); t != "" {
		out = append(out, t)
	}
	for _, cookieName := range []string{"__Secure-better-auth.session_token", "better-auth.session_token"} {
		if c, err := r.Cookie(cookieName); err == nil {
			v := strings.TrimSpace(c.Value)
			if v != "" {
				if i := strings.Index(v, "."); i > 0 {
					v = v[:i]
				}
				out = append(out, v)
			}
		}
	}
	if q := strings.TrimSpace(r.URL.Query().Get("token")); q != "" {
		out = append(out, q)
	}
	seen := map[string]bool{}
	uniq := make([]string, 0, len(out))
	for _, v := range out {
		if seen[v] {
			continue
		}
		seen[v] = true
		uniq = append(uniq, v)
	}
	return uniq
}

func BearerToken(r *http.Request) string {
	h := strings.TrimSpace(r.Header.Get("Authorization"))
	if strings.HasPrefix(strings.ToLower(h), "bearer ") {
		return strings.TrimSpace(h[7:])
	}
	return ""
}
