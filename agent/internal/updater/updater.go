package updater

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type Config struct {
	CurrentVersion string
	Repo           string
	Token          string
	AssetsDir      string
	ExecutablePath string
}

type Updater struct {
	cfg    Config
	client *http.Client
}

type Release struct {
	Version     string    `json:"version"`
	PublishedAt time.Time `json:"published_at"`
	AssetName   string    `json:"asset_name"`
	DownloadURL string    `json:"download_url"`
}

type githubRelease struct {
	TagName     string `json:"tag_name"`
	PublishedAt string `json:"published_at"`
	Assets      []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
	} `json:"assets"`
}

func New(cfg Config) *Updater {
	if strings.TrimSpace(cfg.Repo) == "" {
		cfg.Repo = "suryaumapathy2812/aether"
	}
	if strings.TrimSpace(cfg.ExecutablePath) == "" {
		if exe, err := os.Executable(); err == nil {
			cfg.ExecutablePath = exe
		}
	}
	return &Updater{
		cfg: cfg,
		client: &http.Client{
			Timeout: 2 * time.Minute,
		},
	}
}

func (u *Updater) CurrentVersion() string {
	v := strings.TrimSpace(u.cfg.CurrentVersion)
	if v == "" {
		return "dev"
	}
	return strings.TrimPrefix(v, "v")
}

func (u *Updater) Check(ctx context.Context) (Release, bool, error) {
	release, err := u.fetchLatest(ctx)
	if err != nil {
		return Release{}, false, err
	}
	available := isNewerVersion(strings.TrimPrefix(release.Version, "v"), u.CurrentVersion())
	return release, available, nil
}

func (u *Updater) ApplyLatest(ctx context.Context) (Release, error) {
	release, err := u.fetchLatest(ctx)
	if err != nil {
		return Release{}, err
	}
	tmpDir, err := os.MkdirTemp("", "aether-agent-update-*")
	if err != nil {
		return Release{}, err
	}
	defer os.RemoveAll(tmpDir)

	archivePath := filepath.Join(tmpDir, "agent-update.tar.gz")
	if err := u.downloadTo(ctx, release.DownloadURL, archivePath); err != nil {
		return Release{}, err
	}
	if err := u.applyArchive(archivePath); err != nil {
		return Release{}, err
	}
	return release, nil
}

func (u *Updater) RestartSelf() error {
	exePath := strings.TrimSpace(u.cfg.ExecutablePath)
	if exePath == "" {
		return fmt.Errorf("executable path is empty")
	}
	return syscall.Exec(exePath, os.Args, os.Environ())
}

func (u *Updater) fetchLatest(ctx context.Context) (Release, error) {
	apiURL := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", strings.TrimSpace(u.cfg.Repo))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return Release{}, err
	}
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("User-Agent", "aether-agent-updater")
	if strings.TrimSpace(u.cfg.Token) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(u.cfg.Token))
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return Release{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return Release{}, fmt.Errorf("github latest release request failed: %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var gh githubRelease
	if err := json.NewDecoder(resp.Body).Decode(&gh); err != nil {
		return Release{}, err
	}
	assetName := expectedAssetName(strings.TrimPrefix(gh.TagName, "v"))
	for _, a := range gh.Assets {
		if strings.EqualFold(strings.TrimSpace(a.Name), assetName) {
			publishedAt, _ := time.Parse(time.RFC3339, gh.PublishedAt)
			return Release{
				Version:     strings.TrimSpace(gh.TagName),
				PublishedAt: publishedAt,
				AssetName:   a.Name,
				DownloadURL: a.BrowserDownloadURL,
			}, nil
		}
	}
	return Release{}, fmt.Errorf("release asset %q not found", assetName)
}

func (u *Updater) downloadTo(ctx context.Context, srcURL, destPath string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, srcURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "aether-agent-updater")
	if strings.TrimSpace(u.cfg.Token) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(u.cfg.Token))
	}

	resp, err := u.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("release download failed: %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		return err
	}
	return out.Chmod(0o644)
}

func (u *Updater) applyArchive(archivePath string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	tmpDir, err := os.MkdirTemp("", "aether-agent-apply-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		name := filepath.Clean(hdr.Name)
		if name == "." || strings.HasPrefix(name, "..") {
			continue
		}
		target := filepath.Join(tmpDir, name)
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, os.FileMode(hdr.Mode))
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			if err := out.Close(); err != nil {
				return err
			}
		}
	}

	newBinary := filepath.Join(tmpDir, "agent-server")
	if _, err := os.Stat(newBinary); err != nil {
		return fmt.Errorf("update archive missing agent-server binary")
	}
	if err := os.Chmod(newBinary, 0o755); err != nil {
		return err
	}
	if err := replaceFileAtomic(newBinary, u.cfg.ExecutablePath, 0o755); err != nil {
		return err
	}

	if err := u.applyBuiltinAssets(tmpDir); err != nil {
		return err
	}
	return nil
}

func (u *Updater) applyBuiltinAssets(extractedRoot string) error {
	assetsDir := strings.TrimSpace(u.cfg.AssetsDir)
	if assetsDir == "" {
		return nil
	}

	newPrompt := filepath.Join(extractedRoot, "assets", "PROMPT.md")
	if fi, err := os.Stat(newPrompt); err == nil && !fi.IsDir() {
		if err := replaceFileAtomic(newPrompt, filepath.Join(assetsDir, "PROMPT.md"), 0o644); err != nil {
			return err
		}
	}

	if err := replaceDir(filepath.Join(extractedRoot, "assets", "skills", "builtin"), filepath.Join(assetsDir, "skills", "builtin")); err != nil {
		return err
	}
	if err := replaceDir(filepath.Join(extractedRoot, "assets", "plugins", "builtin"), filepath.Join(assetsDir, "plugins", "builtin")); err != nil {
		return err
	}
	return nil
}

func replaceDir(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	if !info.IsDir() {
		return nil
	}

	parent := filepath.Dir(dst)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp := dst + ".new"
	_ = os.RemoveAll(tmp)
	if err := copyDirRecursive(src, tmp); err != nil {
		_ = os.RemoveAll(tmp)
		return err
	}
	backup := dst + ".old"
	_ = os.RemoveAll(backup)
	if _, err := os.Stat(dst); err == nil {
		if err := os.Rename(dst, backup); err != nil {
			_ = os.RemoveAll(tmp)
			return err
		}
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Rename(backup, dst)
		_ = os.RemoveAll(tmp)
		return err
	}
	_ = os.RemoveAll(backup)
	return nil
}

func copyDirRecursive(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode().Perm())
		}
		in, err := os.Open(path)
		if err != nil {
			return err
		}
		defer in.Close()
		out, err := os.OpenFile(target, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, info.Mode().Perm())
		if err != nil {
			return err
		}
		if _, err := io.Copy(out, in); err != nil {
			out.Close()
			return err
		}
		return out.Close()
	})
}

func replaceFileAtomic(src, dst string, mode os.FileMode) error {
	if strings.TrimSpace(dst) == "" {
		return fmt.Errorf("target path is empty")
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	tmp := dst + ".new"
	if err := copyFile(src, tmp, mode); err != nil {
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func copyFile(src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Close(); err != nil {
		return err
	}
	return os.Chmod(dst, mode)
}

func expectedAssetName(version string) string {
	v := strings.TrimPrefix(strings.TrimSpace(version), "v")
	if v == "" {
		v = "latest"
	}
	return "aether-agent-v" + v + "-" + runtime.GOOS + "-" + runtime.GOARCH + ".tar.gz"
}

func isNewerVersion(latest, current string) bool {
	l := parseVersionTriplet(latest)
	c := parseVersionTriplet(current)
	for i := 0; i < 3; i++ {
		if l[i] > c[i] {
			return true
		}
		if l[i] < c[i] {
			return false
		}
	}
	return false
}

func parseVersionTriplet(v string) [3]int {
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	parts := strings.Split(v, ".")
	out := [3]int{0, 0, 0}
	for i := 0; i < len(parts) && i < 3; i++ {
		n, err := strconv.Atoi(strings.TrimSpace(parts[i]))
		if err == nil && n >= 0 {
			out[i] = n
		}
	}
	return out
}
