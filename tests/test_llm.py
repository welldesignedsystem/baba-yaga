"""Tests for llm module."""

import os
import unittest
from unittest.mock import patch

from botocore.exceptions import ClientError

from llm import (
    BedrockConfig,
    GroqConfig,
    OpenRouterConfig,
    bedrock_chat_model,
    groq_chat_model,
    openrouter_chat_model,
)


# ── OpenRouter ──────────────────────────────────────────────

class TestOpenRouterConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "OPENROUTER_API_KEY": "test-key",
        "OPENROUTER_MODEL": "test-model",
        "OPENROUTER_SITE_URL": "https://example.com",
        "OPENROUTER_APP_NAME": "test-app",
    }, clear=True)
    def test_reads_from_env(self):
        cfg = OpenRouterConfig()
        self.assertEqual(cfg.api_key, "test-key")
        self.assertEqual(cfg.model, "test-model")
        self.assertEqual(cfg.site_url, "https://example.com")
        self.assertEqual(cfg.app_name, "test-app")

    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_default_model_fallback(self):
        cfg = OpenRouterConfig()
        self.assertEqual(cfg.model, "anthropic/claude-3.5-sonnet")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        with self.assertRaises(KeyError):
            OpenRouterConfig()


class TestOpenRouterChatModel(unittest.TestCase):
    @patch("llm.ChatOpenAI")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True)
    def test_defaults(self, mock_chat):
        openrouter_chat_model()
        mock_chat.assert_called_once_with(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.0,
            max_tokens=None,
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "", "X-Title": ""},
        )

    @patch("llm.ChatOpenAI")
    def test_explicit_config(self, mock_chat):
        cfg = OpenRouterConfig(
            api_key="explicit-key", model="explicit-model",
            site_url="https://explicit.com", app_name="explicit-app",
        )
        openrouter_chat_model(config=cfg)
        mock_chat.assert_called_once_with(
            model="explicit-model", temperature=0.0, max_tokens=None,
            api_key="explicit-key", base_url="https://openrouter.ai/api/v1",
            default_headers={"HTTP-Referer": "https://explicit.com", "X-Title": "explicit-app"},
        )


# ── Bedrock ─────────────────────────────────────────────────

class TestBedrockConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "BEDROCK_MODEL": "test-model",
        "AWS_REGION": "test-region",
        "AWS_PROFILE": "test-profile",
    }, clear=True)
    def test_reads_from_env(self):
        cfg = BedrockConfig()
        self.assertEqual(cfg.model, "test-model")
        self.assertEqual(cfg.region_name, "test-region")
        self.assertEqual(cfg.profile_name, "test-profile")

    @patch.dict(os.environ, {}, clear=True)
    def test_defaults(self):
        cfg = BedrockConfig()
        self.assertEqual(cfg.model, "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(cfg.region_name, "us-east-1")
        self.assertEqual(cfg.profile_name, "default")


class TestBedrockChatModel(unittest.TestCase):
    @patch("llm.ChatBedrock")
    @patch.dict(os.environ, {
        "BEDROCK_MODEL": "test-model",
        "AWS_REGION": "test-region",
        "AWS_PROFILE": "test-profile",
    }, clear=True)
    def test_defaults(self, mock_bedrock):
        bedrock_chat_model()
        mock_bedrock.assert_called_once_with(
            model="test-model", temperature=0.0, max_tokens=None,
            region_name="test-region", credentials_profile_name="test-profile",
        )

    @patch("llm.ChatBedrock")
    def test_explicit_config(self, mock_bedrock):
        cfg = BedrockConfig(model="custom-model", region_name="custom-region", profile_name="custom-profile")
        bedrock_chat_model(config=cfg)
        mock_bedrock.assert_called_once_with(
            model="custom-model", temperature=0.0, max_tokens=None,
            region_name="custom-region", credentials_profile_name="custom-profile",
        )


# ── Groq ────────────────────────────────────────────────────

class TestGroqConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "GROQ_API_KEY": "test-key",
        "GROQ_MODEL": "test-model",
    }, clear=True)
    def test_reads_from_env(self):
        cfg = GroqConfig()
        self.assertEqual(cfg.api_key, "test-key")
        self.assertEqual(cfg.model, "test-model")

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True)
    def test_default_model_fallback(self):
        cfg = GroqConfig()
        self.assertEqual(cfg.model, "llama-3.3-70b-versatile")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises(self):
        with self.assertRaises(KeyError):
            GroqConfig()


class TestGroqChatModel(unittest.TestCase):
    @patch("llm.ChatGroq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}, clear=True)
    def test_defaults(self, mock_groq):
        groq_chat_model()
        mock_groq.assert_called_once_with(
            model="llama-3.3-70b-versatile", temperature=0.0, max_tokens=None, api_key="test-key",
        )

    @patch("llm.ChatGroq")
    def test_explicit_config(self, mock_groq):
        cfg = GroqConfig(api_key="explicit-key", model="explicit-model")
        groq_chat_model(config=cfg)
        mock_groq.assert_called_once_with(
            model="explicit-model", temperature=0.0, max_tokens=None, api_key="explicit-key",
        )


# ── Integration tests (actual API calls) ────────────────────

class TestCapitalOfFrance(unittest.TestCase):
    """Asks each configured provider for the capital of France."""

    def _check(self, model, env_var, ignore_client_error=False):
        if not os.environ.get(env_var):
            self.skipTest(f"{env_var} not set")
        try:
            response = model.invoke("What is the capital of France? Answer in one word.")
        except ClientError as e:
            if ignore_client_error:
                self.skipTest(str(e))
            raise
        self.assertIn("paris", response.content.strip().lower())

    def test_openrouter(self):
        self._check(openrouter_chat_model(temperature=0.0), "OPENROUTER_API_KEY")

    def test_bedrock(self):
        self._check(bedrock_chat_model(temperature=0.0), "BEDROCK_MODEL", ignore_client_error=True)

    def test_groq(self):
        self._check(groq_chat_model(temperature=0.0), "GROQ_API_KEY")
