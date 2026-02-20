"""Tests for Vobiz plugin tools."""

import sys
from pathlib import Path

import pytest

# Add plugin directory to path for imports
plugin_dir = Path(__file__).parent.parent / "plugins" / "vobiz"
sys.path.insert(0, str(plugin_dir.parent.parent))

from plugins.vobiz.tools import (
    GetUserPhoneNumberTool,
    MakePhoneCallTool,
    OutboundCallResult,
)


class TestMakePhoneCallTool:
    """Tests for MakePhoneCallTool."""

    def test_tool_attributes(self) -> None:
        """Test tool has correct attributes."""
        tool = MakePhoneCallTool()
        assert tool.name == "make_phone_call"
        assert "phone call" in tool.description.lower()
        assert tool.status_text == "Initiating phone call..."
        assert len(tool.parameters) == 3

    @pytest.mark.asyncio
    async def test_execute_missing_config(self) -> None:
        """Test execute fails when config is missing."""
        tool = MakePhoneCallTool()
        tool._context = {}  # No config

        result = await tool.execute()

        assert result.error
        assert "not configured" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_missing_from_number(self) -> None:
        """Test execute fails when from_number is missing."""
        tool = MakePhoneCallTool()
        tool._context = {
            "auth_id": "test_id",
            "auth_token": "test_token",
            "base_url": "http://localhost:8000",
        }

        result = await tool.execute()

        assert result.error
        assert "phone number" in result.output.lower()

    @pytest.mark.asyncio
    async def test_execute_no_destination(self) -> None:
        """Test execute fails when no destination number."""
        tool = MakePhoneCallTool()
        tool._context = {
            "auth_id": "test_id",
            "auth_token": "test_token",
            "from_number": "919876543210",
            "base_url": "http://localhost:8000",
        }

        result = await tool.execute()

        assert result.error
        assert "no phone number" in result.output.lower()


class TestGetUserPhoneNumberTool:
    """Tests for GetUserPhoneNumberTool."""

    def test_tool_attributes(self) -> None:
        """Test tool has correct attributes."""
        tool = GetUserPhoneNumberTool()
        assert tool.name == "get_user_phone_number"
        assert "phone number" in tool.description.lower()
        assert len(tool.parameters) == 0

    @pytest.mark.asyncio
    async def test_execute_with_phone_number(self) -> None:
        """Test execute returns configured phone number."""
        tool = GetUserPhoneNumberTool()
        tool._context = {"user_phone_number": "+919123456789"}

        result = await tool.execute()

        assert not result.error
        assert "+919123456789" in result.output

    @pytest.mark.asyncio
    async def test_execute_without_phone_number(self) -> None:
        """Test execute returns message when no phone number configured."""
        tool = GetUserPhoneNumberTool()
        tool._context = {}

        result = await tool.execute()

        assert not result.error
        assert "no phone number" in result.output.lower()


class TestOutboundCallResult:
    """Tests for OutboundCallResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = OutboundCallResult(success=True, call_uuid="test-uuid")
        assert result.success
        assert result.call_uuid == "test-uuid"
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failure result."""
        result = OutboundCallResult(success=False, error="Test error")
        assert not result.success
        assert result.error == "Test error"
        assert result.call_uuid is None
