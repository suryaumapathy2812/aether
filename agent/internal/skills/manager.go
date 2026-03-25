package skills

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

type Manager struct {
	mu          sync.RWMutex
	index       map[string]SkillMeta
	dirs        map[SourceType][]string
	userDir     string
	externalDir string
	httpClient  HTTPDoer
	rawBaseURL  string
	apiBaseURL  string
}

func NewManager(opts ManagerOptions) *Manager {
	httpClient := opts.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	rawBase := strings.TrimRight(opts.RawBaseURL, "/")
	if rawBase == "" {
		rawBase = "https://raw.githubusercontent.com"
	}
	apiBase := strings.TrimRight(opts.SkillsAPI, "/")
	if apiBase == "" {
		apiBase = "https://skills.sh/api"
	}

	m := &Manager{
		index:       map[string]SkillMeta{},
		dirs:        map[SourceType][]string{},
		userDir:     opts.UserDir,
		externalDir: opts.ExternalDir,
		httpClient:  httpClient,
		rawBaseURL:  rawBase,
		apiBaseURL:  apiBase,
	}

	for _, dir := range opts.BuiltinDirs {
		_ = m.AttachDirectory(dir, SourceBuiltin)
	}
	if opts.UserDir != "" {
		_ = m.AttachDirectory(opts.UserDir, SourceUser)
	}
	if opts.ExternalDir != "" {
		_ = m.AttachDirectory(opts.ExternalDir, SourceExternal)
	}

	return m
}

func (m *Manager) SearchMarketplace(ctx context.Context, query string, limit int) (MarketplaceSearchResult, error) {
	q := strings.TrimSpace(query)
	if q == "" {
		return MarketplaceSearchResult{}, fmt.Errorf("%w: query is required", ErrInvalidSource)
	}
	if limit <= 0 || limit > 50 {
		limit = 10
	}

	requestURL := fmt.Sprintf("%s/search?q=%s&limit=%d", m.apiBaseURL, url.QueryEscape(q), limit)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return MarketplaceSearchResult{}, err
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return MarketplaceSearchResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return MarketplaceSearchResult{}, fmt.Errorf("marketplace search failed: status %d", resp.StatusCode)
	}

	var raw struct {
		Query      string `json:"query"`
		SearchType string `json:"searchType"`
		Skills     []struct {
			ID       string `json:"id"`
			SkillID  string `json:"skillId"`
			Name     string `json:"name"`
			Installs int    `json:"installs"`
			Source   string `json:"source"`
		} `json:"skills"`
		Count      int `json:"count"`
		DurationMS int `json:"duration_ms"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return MarketplaceSearchResult{}, err
	}

	out := MarketplaceSearchResult{
		Query:      raw.Query,
		SearchType: raw.SearchType,
		Skills:     make([]MarketplaceSkill, 0, len(raw.Skills)),
		Count:      raw.Count,
		DurationMS: raw.DurationMS,
	}
	for _, s := range raw.Skills {
		out.Skills = append(out.Skills, MarketplaceSkill{
			ID:       strings.TrimSpace(s.ID),
			SkillID:  strings.TrimSpace(s.SkillID),
			Name:     strings.TrimSpace(s.Name),
			Installs: s.Installs,
			Source:   strings.TrimSpace(s.Source),
		})
	}
	if out.Count == 0 {
		out.Count = len(out.Skills)
	}
	return out, nil
}

func (m *Manager) AttachDirectory(path string, source SourceType) error {
	if source != SourceBuiltin && source != SourceUser && source != SourceExternal {
		return ErrInvalidSource
	}
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("%w: empty path", ErrInvalidSource)
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, existing := range m.dirs[source] {
		if existing == abs {
			return nil
		}
	}
	m.dirs[source] = append(m.dirs[source], abs)
	return nil
}

func (m *Manager) Discover(ctx context.Context) (int, error) {
	_ = ctx
	temp := map[string]SkillMeta{}

	sourceOrder := []SourceType{SourceBuiltin, SourceUser, SourceExternal}
	for _, source := range sourceOrder {
		for _, dir := range m.snapshotDirs(source) {
			count, err := scanDir(dir, source, temp)
			if err != nil {
				return 0, err
			}
			_ = count
		}
	}

	m.mu.Lock()
	m.index = temp
	m.mu.Unlock()

	return len(temp), nil
}

func (m *Manager) Refresh(ctx context.Context) error {
	_, err := m.Discover(ctx)
	return err
}

func (m *Manager) List() []SkillMeta {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]SkillMeta, 0, len(m.index))
	for _, meta := range m.index {
		out = append(out, meta)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (m *Manager) Get(name string) (SkillMeta, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	meta, ok := m.index[normalizeName(name)]
	return meta, ok
}

func (m *Manager) Read(name string) (string, error) {
	meta, ok := m.Get(name)
	if !ok {
		return "", ErrNotFound
	}
	raw, err := os.ReadFile(meta.Location)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrNotFound
		}
		return "", err
	}
	parsed, err := parseSkillMarkdown(string(raw))
	if err != nil {
		return "", err
	}
	return parsed.Body, nil
}

func (m *Manager) ReadRaw(name string) (string, error) {
	meta, ok := m.Get(name)
	if !ok {
		return "", ErrNotFound
	}
	raw, err := os.ReadFile(meta.Location)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrNotFound
		}
		return "", err
	}
	return string(raw), nil
}

func (m *Manager) Search(query string) []SkillMeta {
	query = normalizeName(query)
	if query == "" {
		return m.List()
	}
	words := splitWords(query)

	type scored struct {
		score int
		meta  SkillMeta
	}
	scoredMatches := []scored{}

	for _, meta := range m.List() {
		hay := normalizeName(meta.Name + " " + meta.Description)
		hayWords := splitWords(hay)
		score := overlap(words, hayWords)
		if score > 0 {
			scoredMatches = append(scoredMatches, scored{score: score, meta: meta})
		}
	}

	sort.Slice(scoredMatches, func(i, j int) bool {
		if scoredMatches[i].score == scoredMatches[j].score {
			return scoredMatches[i].meta.Name < scoredMatches[j].meta.Name
		}
		return scoredMatches[i].score > scoredMatches[j].score
	})

	out := make([]SkillMeta, 0, len(scoredMatches))
	for _, item := range scoredMatches {
		out = append(out, item.meta)
	}
	return out
}

func (m *Manager) Create(name, description, content string) (SkillMeta, error) {
	if strings.TrimSpace(description) == "" || strings.TrimSpace(content) == "" {
		return SkillMeta{}, fmt.Errorf("%w: description and content are required", ErrInvalidSkill)
	}
	if strings.TrimSpace(m.userDir) == "" {
		return SkillMeta{}, fmt.Errorf("%w: user directory is not configured", ErrInvalidSource)
	}

	safe := sanitizeName(name)
	if _, ok := m.Get(safe); ok {
		return SkillMeta{}, fmt.Errorf("%w: %s", ErrDuplicateName, safe)
	}

	skillDir := filepath.Join(m.userDir, safe)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		return SkillMeta{}, err
	}
	path := filepath.Join(skillDir, "SKILL.md")
	if err := os.WriteFile(path, []byte(buildSkillMarkdown(safe, description, content)), 0o644); err != nil {
		return SkillMeta{}, err
	}

	meta := SkillMeta{
		Name:        safe,
		Description: strings.TrimSpace(description),
		Location:    path,
		Source:      SourceUser,
	}
	if err := m.register(meta); err != nil {
		return SkillMeta{}, err
	}
	return meta, nil
}

func (m *Manager) Remove(name string) error {
	meta, ok := m.Get(name)
	if !ok {
		return ErrNotFound
	}
	if meta.Source == SourceBuiltin {
		return ErrProtected
	}

	if err := os.Remove(meta.Location); err != nil && !os.IsNotExist(err) {
		return err
	}
	_ = os.RemoveAll(filepath.Dir(meta.Location))

	m.mu.Lock()
	delete(m.index, normalizeName(meta.Name))
	m.mu.Unlock()
	return nil
}

func (m *Manager) InstallFromSource(ctx context.Context, source string) (InstallResult, error) {
	owner, repo, skillName, err := parseExternalSource(source)
	if err != nil {
		return InstallResult{}, err
	}
	if strings.TrimSpace(m.externalDir) == "" {
		return InstallResult{}, fmt.Errorf("%w: external directory is not configured", ErrInvalidSource)
	}

	candidates := buildRawCandidates(m.rawBaseURL, owner, repo, skillName)
	remoteURL := ""
	var body []byte
	var found bool
	for _, candidate := range candidates {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, candidate, nil)
		if err != nil {
			return InstallResult{}, err
		}
		resp, err := m.httpClient.Do(req)
		if err != nil {
			return InstallResult{}, err
		}
		if resp.StatusCode == http.StatusNotFound {
			_ = resp.Body.Close()
			continue
		}
		if resp.StatusCode != http.StatusOK {
			_ = resp.Body.Close()
			return InstallResult{}, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
		}
		payload, err := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if err != nil {
			return InstallResult{}, err
		}
		remoteURL = candidate
		body = payload
		found = true
		break
	}
	if !found {
		return InstallResult{}, ErrNotFound
	}

	parsed, err := parseSkillMarkdown(string(body))
	if err != nil {
		return InstallResult{}, err
	}

	safe := sanitizeName(parsed.Name)
	if _, ok := m.Get(safe); ok {
		return InstallResult{}, fmt.Errorf("%w: %s", ErrDuplicateName, safe)
	}

	skillDir := filepath.Join(m.externalDir, safe)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		return InstallResult{}, err
	}
	path := filepath.Join(skillDir, "SKILL.md")
	if err := os.WriteFile(path, body, 0o644); err != nil {
		return InstallResult{}, err
	}

	meta := SkillMeta{
		Name:        safe,
		Description: parsed.Description,
		Location:    path,
		Source:      SourceExternal,
	}
	if err := m.register(meta); err != nil {
		return InstallResult{}, err
	}

	return InstallResult{
		Source:     source,
		RemoteURL:  remoteURL,
		Installed:  meta,
		Downloaded: true,
	}, nil
}

func (m *Manager) snapshotDirs(source SourceType) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	dirs := m.dirs[source]
	out := make([]string, len(dirs))
	copy(out, dirs)
	return out
}

func (m *Manager) register(meta SkillMeta) error {
	key := normalizeName(meta.Name)
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.index[key]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateName, meta.Name)
	}
	m.index[key] = meta
	return nil
}

func scanDir(root string, source SourceType, acc map[string]SkillMeta) (int, error) {
	stat, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	if !stat.IsDir() {
		return 0, nil
	}

	count := 0
	err = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() || d.Name() != "SKILL.md" {
			return nil
		}

		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		parsed, err := parseSkillMarkdown(string(raw))
		if err != nil {
			return err
		}
		key := normalizeName(parsed.Name)
		if _, exists := acc[key]; exists {
			return fmt.Errorf("%w: %s", ErrDuplicateName, parsed.Name)
		}
		acc[key] = SkillMeta{
			Name:        parsed.Name,
			Description: parsed.Description,
			AlwaysLoad:  parsed.AlwaysLoad,
			Integration: parsed.Integration,
			Location:    path,
			Source:      source,
		}
		count++
		return nil
	})
	if err != nil {
		return 0, err
	}
	return count, nil
}

func splitWords(s string) map[string]struct{} {
	words := map[string]struct{}{}
	for _, w := range strings.Fields(s) {
		w = strings.Trim(w, "-_.,:;!?()[]{}\"'")
		if w == "" {
			continue
		}
		words[w] = struct{}{}
	}
	return words
}

func overlap(a, b map[string]struct{}) int {
	n := 0
	for w := range a {
		if _, ok := b[w]; ok {
			n++
		}
	}
	return n
}
