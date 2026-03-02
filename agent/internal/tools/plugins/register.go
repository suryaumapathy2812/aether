package plugins

import (
	"strings"

	coreplugins "github.com/suryaumapathy2812/core-ai/agent/internal/plugins"
	"github.com/suryaumapathy2812/core-ai/agent/internal/tools"
)

func RegisterAvailable(reg *tools.Registry, manager *coreplugins.Manager) error {
	if reg == nil || manager == nil {
		return nil
	}
	constructors := classConstructors()
	for _, meta := range manager.List() {
		manifest, err := manager.ReadManifest(meta.Name)
		if err != nil {
			continue
		}
		for _, rawTool := range manifest.Tools {
			className, _ := rawTool["class"].(string)
			className = strings.TrimSpace(className)
			if className == "" {
				continue
			}
			ctor, ok := constructors[className]
			if !ok {
				continue
			}
			if err := reg.Register(ctor(), meta.Name); err != nil {
				return err
			}
		}
	}
	return nil
}

func classConstructors() map[string]func() tools.Tool {
	return map[string]func() tools.Tool{
		"CurrentWeatherTool":             func() tools.Tool { return &CurrentWeatherTool{} },
		"ForecastTool":                   func() tools.Tool { return &ForecastTool{} },
		"WikipediaSearchTool":            func() tools.Tool { return &WikipediaSearchTool{} },
		"WikipediaGetArticleTool":        func() tools.Tool { return &WikipediaGetArticleTool{} },
		"WebSearchTool":                  func() tools.Tool { return &WebSearchTool{} },
		"NewsSearchTool":                 func() tools.Tool { return &NewsSearchTool{} },
		"LLMContextSearchTool":           func() tools.Tool { return &LLMContextSearchTool{} },
		"LocalSearchTool":                func() tools.Tool { return &LocalSearchTool{} },
		"FetchFeedTool":                  func() tools.Tool { return &FetchFeedTool{} },
		"GetItemContentTool":             func() tools.Tool { return &GetItemContentTool{} },
		"ListSubscribedFeedsTool":        func() tools.Tool { return &ListSubscribedFeedsTool{} },
		"AddFeedTool":                    func() tools.Tool { return &AddFeedTool{} },
		"GetHackerNewsTopTool":           func() tools.Tool { return &GetHackerNewsTopTool{} },
		"SearchHackerNewsTool":           func() tools.Tool { return &SearchHackerNewsTool{} },
		"WolframQueryTool":               func() tools.Tool { return &WolframQueryTool{} },
		"CurrencyConvertTool":            func() tools.Tool { return &CurrencyConvertTool{} },
		"ListUnreadTool":                 func() tools.Tool { return &ListUnreadTool{} },
		"ReadEmailTool":                  func() tools.Tool { return &ReadEmailTool{} },
		"SendReplyTool":                  func() tools.Tool { return &SendReplyTool{} },
		"SendEmailTool":                  func() tools.Tool { return &SendEmailTool{} },
		"CreateDraftTool":                func() tools.Tool { return &CreateDraftTool{} },
		"ArchiveEmailTool":               func() tools.Tool { return &ArchiveEmailTool{} },
		"SearchEmailTool":                func() tools.Tool { return &SearchEmailTool{} },
		"ReplyAllTool":                   func() tools.Tool { return &ReplyAllTool{} },
		"GetThreadTool":                  func() tools.Tool { return &GetThreadTool{} },
		"TrashEmailTool":                 func() tools.Tool { return &TrashEmailTool{} },
		"MarkReadTool":                   func() tools.Tool { return &MarkReadTool{} },
		"MarkUnreadTool":                 func() tools.Tool { return &MarkUnreadTool{} },
		"ListLabelsTool":                 func() tools.Tool { return &ListLabelsTool{} },
		"AddLabelTool":                   func() tools.Tool { return &AddLabelTool{} },
		"RemoveLabelTool":                func() tools.Tool { return &RemoveLabelTool{} },
		"RefreshGmailTokenTool":          func() tools.Tool { return &RefreshGmailTokenTool{} },
		"SetupGmailWatchTool":            func() tools.Tool { return &SetupGmailWatchTool{} },
		"UpcomingEventsTool":             func() tools.Tool { return &UpcomingEventsTool{} },
		"SearchEventsTool":               func() tools.Tool { return &SearchEventsTool{} },
		"CreateEventTool":                func() tools.Tool { return &CreateEventTool{} },
		"GetEventTool":                   func() tools.Tool { return &GetEventTool{} },
		"RefreshGoogleCalendarTokenTool": func() tools.Tool { return &RefreshGoogleCalendarTokenTool{} },
		"SetupCalendarWatchTool":         func() tools.Tool { return &SetupCalendarWatchTool{} },
		"SearchContactsTool":             func() tools.Tool { return &SearchContactsTool{} },
		"GetContactTool":                 func() tools.Tool { return &GetContactTool{} },
		"RefreshGoogleContactsTokenTool": func() tools.Tool { return &RefreshGoogleContactsTokenTool{} },
		"SearchDriveTool":                func() tools.Tool { return &SearchDriveTool{} },
		"ListDriveFilesTool":             func() tools.Tool { return &ListDriveFilesTool{} },
		"GetFileInfoTool":                func() tools.Tool { return &GetFileInfoTool{} },
		"ReadFileContentTool":            func() tools.Tool { return &ReadFileContentTool{} },
		"ListSharedDrivesTool":           func() tools.Tool { return &ListSharedDrivesTool{} },
		"CreateDocumentTool":             func() tools.Tool { return &CreateDocumentTool{} },
		"CreateSpreadsheetTool":          func() tools.Tool { return &CreateSpreadsheetTool{} },
		"CreatePresentationTool":         func() tools.Tool { return &CreatePresentationTool{} },
		"ShareFileTool":                  func() tools.Tool { return &ShareFileTool{} },
		"ListPermissionsTool":            func() tools.Tool { return &ListPermissionsTool{} },
		"UpdatePermissionTool":           func() tools.Tool { return &UpdatePermissionTool{} },
		"RemoveSharingTool":              func() tools.Tool { return &RemoveSharingTool{} },
		"MakePublicTool":                 func() tools.Tool { return &MakePublicTool{} },
		"MoveFileTool":                   func() tools.Tool { return &MoveFileTool{} },
		"RenameFileTool":                 func() tools.Tool { return &RenameFileTool{} },
		"CopyFileTool":                   func() tools.Tool { return &CopyFileTool{} },
		"CreateFolderTool":               func() tools.Tool { return &CreateFolderTool{} },
		"UpdateDocumentTool":             func() tools.Tool { return &UpdateDocumentTool{} },
		"RefreshGoogleDriveTokenTool":    func() tools.Tool { return &RefreshGoogleDriveTokenTool{} },
		"NowPlayingTool":                 func() tools.Tool { return &NowPlayingTool{} },
		"PlayPauseTool":                  func() tools.Tool { return &PlayPauseTool{} },
		"SkipTrackTool":                  func() tools.Tool { return &SkipTrackTool{} },
		"SearchSpotifyTool":              func() tools.Tool { return &SearchSpotifyTool{} },
		"QueueTrackTool":                 func() tools.Tool { return &QueueTrackTool{} },
		"RecentTracksTool":               func() tools.Tool { return &RecentTracksTool{} },
		"RefreshSpotifyTokenTool":        func() tools.Tool { return &RefreshSpotifyTokenTool{} },
		"SendMessageTool":                func() tools.Tool { return &SendMessageTool{} },
		"SendPhotoTool":                  func() tools.Tool { return &SendPhotoTool{} },
		"GetChatInfoTool":                func() tools.Tool { return &GetChatInfoTool{} },
		"ResolveUsernameTool":            func() tools.Tool { return &ResolveUsernameTool{} },
		"HandleTelegramEventTool":        func() tools.Tool { return &HandleTelegramEventTool{} },
		"SendTypingIndicatorTool":        func() tools.Tool { return &SendTypingIndicatorTool{} },
	}
}
