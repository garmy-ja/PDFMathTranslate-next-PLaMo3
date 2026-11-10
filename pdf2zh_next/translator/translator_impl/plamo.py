"""PLaMo 3 Translate adapter for PDFMathTranslate-next + LM Studio."""

import httpx
import openai

from pdf2zh_next.config.model import SettingsModel
from pdf2zh_next.translator.base_rate_limiter import BaseRateLimiter
from pdf2zh_next.translator.base_translator import BaseTranslator


class PLaMoTranslator(BaseTranslator):
    """Translate text through the PLaMo GGUF model served by LM Studio."""

    # The engine name is also used as the translation-cache engine key.
    name = "plamo"

    _LANGUAGE_ALIASES = {
        "zh": "zh-hans",
        "zh-cn": "zh-hans",
        "zh-sg": "zh-hans",
        "zh-tw": "zh-hant",
        "zh-hk": "zh-hant",
        "zh-mo": "zh-hant",
    }

    _SUPPORTED_LANGUAGES = {
        "ar", "az", "be", "bg", "bn", "ca", "ceb", "de", "en",
        "es", "et", "eu", "fa", "fr", "gu", "he", "hi", "hr",
        "hu", "hy", "id", "it", "ja", "jv", "ka", "kk", "km",
        "ko", "lt", "lv", "ml", "mr", "ms", "my", "ne", "nl",
        "pa", "pl", "ru", "si", "sk", "sl", "sr", "sw", "ta",
        "te", "th", "tl", "ur", "uz", "vi", "zh-hans", "zh-hant",
    }

    def __init__(
        self,
        settings: SettingsModel,
        rate_limiter: BaseRateLimiter,
    ):
        super().__init__(settings, rate_limiter)

        cfg = settings.translate_engine_settings
        self.base_url = cfg.plamo_base_url
        self.api_key = cfg.plamo_api_key or "lm-studio"
        self.model = cfg.plamo_model
        self.timeout = float(cfg.plamo_timeout)
        self.max_tokens = int(cfg.plamo_max_tokens)

        self.target_lang = self._normalize_language(
            self.lang_out, allow_auto=False
        )
        self.source_lang = self._normalize_language(
            self.lang_in, allow_auto=True
        )

        # Include all result-affecting options in the cache identity.
        self.add_cache_impact_parameters("base_url", self.base_url)
        self.add_cache_impact_parameters("model", self.model)
        self.add_cache_impact_parameters("source_lang", self.source_lang)
        self.add_cache_impact_parameters("target_lang", self.target_lang)
        self.add_cache_impact_parameters("max_tokens", self.max_tokens)
        self.add_cache_impact_parameters("prompt_format", "plamo-translate-v1")

        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=3,
            http_client=httpx.Client(
                limits=httpx.Limits(
                    max_connections=None,
                    max_keepalive_connections=None,
                )
            ),
        )

    @classmethod
    def _normalize_language(cls, language: str, *, allow_auto: bool) -> str | None:
        """Convert PDFMathTranslate language codes to PLaMo language codes."""
        value = (language or "").strip().replace("_", "-").lower()

        if allow_auto and value in {"", "auto", "automatic", "auto-detect"}:
            return None

        value = cls._LANGUAGE_ALIASES.get(value, value)
        if value not in cls._SUPPORTED_LANGUAGES:
            raise ValueError(
                f"PLaMo 3 Translate does not support language code: {language!r}. "
                "Check the language code selected in PDFMathTranslate-next."
            )
        return value

    def do_translate(self, text, rate_limit_params: dict = None) -> str:
        """Send only the source text as user content; put options in PLaMo header."""
        header_lines = [
            "[plamo-translate]",
            f"target_lang={self.target_lang}",
        ]
        if self.source_lang:
            header_lines.append(f"source_lang={self.source_lang}")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "\n".join(header_lines),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            temperature=0,
            max_tokens=self.max_tokens,
            stop=["<|plamo:tag|>", "<|plamo:bos|>"],
        )

        if not response.choices:
            raise RuntimeError("LM Studio returned no completion choices")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("PLaMo returned an empty translation")

        return self._remove_cot_content(content).strip()
