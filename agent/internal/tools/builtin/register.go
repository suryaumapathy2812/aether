package builtin

import "github.com/suryaumapathy/core-ai/agent/internal/tools"

func RegisterCore(reg *tools.Registry) error {
	core := []tools.Tool{
		&ListDirectoryTool{},
		&ReadFileTool{},
		&WriteFileTool{},
		&WorldTimeTool{},
		&RunCommandTool{},
		&ScheduleReminderTool{},
		&SaveMemoryTool{},
		&SearchMemoryTool{},
	}
	for _, t := range core {
		if err := reg.Register(t, ""); err != nil {
			return err
		}
	}
	return nil
}

func RegisterManagement(reg *tools.Registry) error {
	mgmt := []tools.Tool{
		&SearchSkillTool{},
		&ReadSkillTool{},
		&CreateSkillTool{},
		&InstallSkillTool{},
		&RemoveSkillTool{},
		&ListPluginsTool{},
		&ReadPluginManifestTool{},
		&InstallPluginTool{},
		&EnablePluginTool{},
		&DisablePluginTool{},
		&SetPluginConfigTool{},
		&ListJobsTool{},
		&ScheduleJobTool{},
		&CancelJobTool{},
		&PauseJobTool{},
		&ResumeJobTool{},
		&DelegateTaskTool{},
		&GetTaskStatusTool{},
		&ListTasksTool{},
		&CancelTaskTool{},
		&GetTaskResultTool{},
		&RequestHumanApprovalTool{},
		&ResumeTaskTool{},
		&ListPendingApprovalsTool{},
		&ApproveTaskTool{},
		&RejectTaskTool{},
	}
	for _, t := range mgmt {
		if err := reg.Register(t, ""); err != nil {
			return err
		}
	}
	return nil
}

func RegisterAll(reg *tools.Registry) error {
	if err := RegisterCore(reg); err != nil {
		return err
	}
	return RegisterManagement(reg)
}
