package plugins

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/suryaumapathy2812/core-ai/agent/internal/db"
)

type Manager struct {
	mu          sync.RWMutex
	index       map[string]PluginManifest
	dirs        map[SourceType][]string
	userDir     string
	externalDir string
	httpClient  HTTPDoer
	rawBaseURL  string
	stateStore  *db.Store
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

	m := &Manager{
		index:       map[string]PluginManifest{},
		dirs:        map[SourceType][]string{},
		userDir:     opts.UserDir,
		externalDir: opts.ExternalDir,
		httpClient:  httpClient,
		rawBaseURL:  rawBase,
		stateStore:  opts.StateStore,
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
	temp := map[string]PluginManifest{}
	for _, source := range []SourceType{SourceBuiltin, SourceUser, SourceExternal} {
		for _, dir := range m.snapshotDirs(source) {
			if err := scanDir(dir, source, temp); err != nil {
				return 0, err
			}
		}
	}

	if err := m.syncStateStore(ctx, temp); err != nil {
		return 0, err
	}

	m.mu.Lock()
	m.index = temp
	m.mu.Unlock()
	return len(temp), nil
}

func (m *Manager) syncStateStore(ctx context.Context, discovered map[string]PluginManifest) error {
	if m.stateStore == nil {
		return nil
	}
	for _, manifest := range discovered {
		rec := db.PluginRecord{
			Name:        manifest.Name,
			DisplayName: manifest.DisplayName,
			Description: manifest.Description,
			Version:     manifest.Version,
			PluginType:  manifest.PluginType,
			Location:    manifest.Location,
			Source:      string(manifest.Source),
			HasSkill:    strings.TrimSpace(manifest.SkillPath) != "",
		}
		if err := m.stateStore.UpsertPlugin(ctx, rec); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manager) Refresh(ctx context.Context) error {
	_, err := m.Discover(ctx)
	return err
}

func (m *Manager) List() []PluginMeta {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]PluginMeta, 0, len(m.index))
	for _, manifest := range m.index {
		out = append(out, toMeta(manifest))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

func (m *Manager) ListEnabled(ctx context.Context) ([]db.PluginRecord, error) {
	if m == nil || m.stateStore == nil {
		return nil, nil
	}
	recs, err := m.stateStore.ListPlugins(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]db.PluginRecord, 0, len(recs))
	for _, rec := range recs {
		if rec.Enabled {
			out = append(out, rec)
		}
	}
	return out, nil
}

func (m *Manager) Get(name string) (PluginMeta, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.index[normalizeName(name)]
	if !ok {
		return PluginMeta{}, false
	}
	return toMeta(p), true
}

func (m *Manager) ReadManifest(name string) (PluginManifest, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	p, ok := m.index[normalizeName(name)]
	if !ok {
		return PluginManifest{}, ErrNotFound
	}
	return p, nil
}

func (m *Manager) RequiredConfigKeys(name string) ([]string, error) {
	m.mu.RLock()
	p, ok := m.index[normalizeName(name)]
	m.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}

	rawFields, ok := p.Auth["config_fields"]
	if !ok {
		return []string{}, nil
	}
	fields, ok := rawFields.([]any)
	if !ok {
		return []string{}, nil
	}
	required := make([]string, 0, len(fields))
	for _, f := range fields {
		entry, ok := f.(map[string]any)
		if !ok {
			continue
		}
		key, _ := entry["key"].(string)
		if strings.TrimSpace(key) == "" {
			continue
		}
		isRequired := false
		switch v := entry["required"].(type) {
		case bool:
			isRequired = v
		case string:
			isRequired = strings.EqualFold(strings.TrimSpace(v), "true")
		}
		if isRequired {
			required = append(required, key)
		}
	}
	return required, nil
}

func (m *Manager) ReadSkill(name string) (string, error) {
	m.mu.RLock()
	p, ok := m.index[normalizeName(name)]
	m.mu.RUnlock()
	if !ok {
		return "", ErrNotFound
	}
	if strings.TrimSpace(p.SkillPath) == "" {
		return "", ErrNotFound
	}
	b, err := os.ReadFile(p.SkillPath)
	if err != nil {
		if os.IsNotExist(err) {
			return "", ErrNotFound
		}
		return "", err
	}
	return string(b), nil
}

func (m *Manager) Search(query string) []PluginMeta {
	query = normalizeName(query)
	if query == "" {
		return m.List()
	}
	words := splitWords(query)
	type scored struct {
		score int
		meta  PluginMeta
	}
	matches := []scored{}
	for _, p := range m.List() {
		hayWords := splitWords(normalizeName(p.Name + " " + p.DisplayName + " " + p.Description + " " + p.PluginType))
		s := overlap(words, hayWords)
		if s > 0 {
			matches = append(matches, scored{score: s, meta: p})
		}
	}
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].score == matches[j].score {
			return matches[i].meta.Name < matches[j].meta.Name
		}
		return matches[i].score > matches[j].score
	})
	out := make([]PluginMeta, 0, len(matches))
	for _, m := range matches {
		out = append(out, m.meta)
	}
	return out
}

func (m *Manager) Remove(name string) error {
	m.mu.RLock()
	p, ok := m.index[normalizeName(name)]
	m.mu.RUnlock()
	if !ok {
		return ErrNotFound
	}
	if p.Source == SourceBuiltin {
		return ErrProtected
	}

	if err := os.Remove(p.Location); err != nil && !os.IsNotExist(err) {
		return err
	}
	if strings.TrimSpace(p.SkillPath) != "" {
		_ = os.Remove(p.SkillPath)
	}
	_ = os.RemoveAll(filepath.Dir(p.Location))

	m.mu.Lock()
	delete(m.index, normalizeName(p.Name))
	m.mu.Unlock()
	return nil
}

func (m *Manager) InstallFromSource(ctx context.Context, source string) (InstallResult, error) {
	owner, repo, pluginPath, err := parseExternalSource(source)
	if err != nil {
		return InstallResult{}, err
	}
	if strings.TrimSpace(m.externalDir) == "" {
		return InstallResult{}, fmt.Errorf("%w: external directory is not configured", ErrInvalidSource)
	}

	manifestURL := buildRawManifestURL(m.rawBaseURL, owner, repo, pluginPath)
	manifestBytes, statusCode, err := m.getURL(ctx, manifestURL)
	if err != nil {
		return InstallResult{}, err
	}
	if statusCode == http.StatusNotFound {
		return InstallResult{}, ErrNotFound
	}
	if statusCode != http.StatusOK {
		return InstallResult{}, fmt.Errorf("unexpected status code: %d", statusCode)
	}

	manifest, err := parseManifest(manifestBytes)
	if err != nil {
		return InstallResult{}, err
	}
	if _, ok := m.Get(manifest.Name); ok {
		return InstallResult{}, fmt.Errorf("%w: %s", ErrDuplicateName, manifest.Name)
	}

	targetDir := filepath.Join(m.externalDir, manifest.Name)
	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return InstallResult{}, err
	}
	manifestPath := filepath.Join(targetDir, "plugin.yaml")
	if err := os.WriteFile(manifestPath, manifestBytes, 0o644); err != nil {
		return InstallResult{}, err
	}

	skillURL := buildRawSkillURL(m.rawBaseURL, owner, repo, pluginPath)
	skillBytes, skillStatus, err := m.getURL(ctx, skillURL)
	skillPath := ""
	if err == nil && skillStatus == http.StatusOK {
		skillPath = filepath.Join(targetDir, "SKILL.md")
		if writeErr := os.WriteFile(skillPath, skillBytes, 0o644); writeErr != nil {
			return InstallResult{}, writeErr
		}
	}

	manifest.Location = manifestPath
	manifest.Source = SourceExternal
	manifest.SkillPath = skillPath
	if err := m.register(manifest); err != nil {
		return InstallResult{}, err
	}

	return InstallResult{
		Source:     source,
		RemoteURL:  manifestURL,
		Installed:  toMeta(manifest),
		Downloaded: true,
	}, nil
}

func (m *Manager) getURL(ctx context.Context, url string) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}
	resp, err := m.httpClient.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, 0, err
	}
	return b, resp.StatusCode, nil
}

func (m *Manager) register(p PluginManifest) error {
	key := normalizeName(p.Name)
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.index[key]; exists {
		return fmt.Errorf("%w: %s", ErrDuplicateName, p.Name)
	}
	m.index[key] = p
	return nil
}

func (m *Manager) snapshotDirs(source SourceType) []string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	dirs := m.dirs[source]
	out := make([]string, len(dirs))
	copy(out, dirs)
	return out
}

func scanDir(root string, source SourceType, acc map[string]PluginManifest) error {
	stat, err := os.Stat(root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if !stat.IsDir() {
		return nil
	}

	return filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() || d.Name() != "plugin.yaml" {
			return nil
		}

		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		manifest, err := parseManifest(raw)
		if err != nil {
			return err
		}
		key := normalizeName(manifest.Name)
		if _, exists := acc[key]; exists {
			return fmt.Errorf("%w: %s", ErrDuplicateName, manifest.Name)
		}

		skillPath := filepath.Join(filepath.Dir(path), "SKILL.md")
		if _, err := os.Stat(skillPath); err == nil {
			manifest.SkillPath = skillPath
		}
		manifest.Location = path
		manifest.Source = source
		acc[key] = manifest
		return nil
	})
}

func toMeta(p PluginManifest) PluginMeta {
	return PluginMeta{
		Name:        p.Name,
		DisplayName: p.DisplayName,
		Description: p.Description,
		Version:     p.Version,
		PluginType:  p.PluginType,
		Location:    p.Location,
		Source:      p.Source,
		HasSkill:    strings.TrimSpace(p.SkillPath) != "",
	}
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
