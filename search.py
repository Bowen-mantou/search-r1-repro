"""提供 DeepSeek Search、Wikipedia、DuckDuckGo 和知乎搜索四个后端。

DeepSeek 后端需要 deepseek_search 包和 API key；其他后端免 key。
"""

import gzip
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx

# DeepSeek search 是可选依赖（需要 API key，已不可用时不影响其他后端）
try:
    from deepseek_search import search as deepseek_search
    from deepseek_search.config import resolve_api_key as _resolve_api_key
except ImportError:
    deepseek_search = None  # type: ignore[assignment]
    _resolve_api_key = None  # type: ignore[assignment]

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]


SEARCH_BACKENDS = ("deepseek", "wikipedia", "zhihu", "bing", "mimo")
SEARCH_ENDPOINT = "https://developer.zhihu.com/api/v1/content/global_search"
WIKIPEDIA_SEARCH_ENDPOINT = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = (
    "agentic-rl-lab/0.1 "
    "(https://github.com/KMnO4-zx/agentic-rl-lab)"
)
DEFAULT_SEARCH_CONCURRENCY = {"deepseek": 16, "wikipedia": 3, "zhihu": 1, "bing": 8, "mimo": 8}
DEFAULT_SEARCH_TIMEOUT = {"deepseek": 60.0, "wikipedia": 15.0, "zhihu": 15.0, "bing": 15.0, "mimo": 30.0}
EVIDENCE_PATTERN = re.compile(
    r"^\[(?:\d+)\]\s*Source:\s*(.*?)\s*\nEvidence:\s*(.*?)"
    r"(?=\n\s*\n\[(?:\d+)\]\s*Source:|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class SearchItem:
    """保存一条搜索证据及其可选来源信息。"""

    title: str
    content: str
    source: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """保存一次搜索的证据或错误信息。"""

    ok: bool
    items: list[SearchItem]
    latency: float
    status: int | None = None
    error: str | None = None


@dataclass
class SearchStats:
    """累计并发搜索请求的运行指标。"""

    backend: str
    requests: int = 0
    successes: int = 0
    timeouts: int = 0
    rate_limits: int = 0
    errors: int = 0
    latency_total: float = 0.0
    credential_failovers: int = 0
    web_search_requests: int = 0
    result_count: int = 0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def metrics(self) -> dict[str, float]:
        """把累计计数转换成便于 SwanLab 记录的单次请求均值。"""
        with self._lock:
            denominator = max(self.requests, 1)
            metrics = {
                "search/success_rate": self.successes / denominator,
                "search/timeout_rate": self.timeouts / denominator,
                "search/429_rate": self.rate_limits / denominator,
                "search/error_rate": self.errors / denominator,
                "search/latency": self.latency_total / denominator,
                "search/results": self.result_count / denominator,
            }
            if self.backend == "deepseek":
                metrics.update(
                    {
                        "search/web_search_requests": (
                            self.web_search_requests / denominator
                        ),
                        "search/input_tokens": self.input_tokens / denominator,
                        "search/cache_read_input_tokens": (
                            self.cache_read_input_tokens / denominator
                        ),
                        "search/output_tokens": self.output_tokens / denominator,
                    }
                )
            if self.backend == "zhihu":
                metrics["search/credential_failover_rate"] = (
                    self.credential_failovers / denominator
                )
            return metrics


@dataclass
class DeepSeekSearchClient:
    """调用受约束的 Evidence 模式，并把返回文本转成工具证据。"""

    api_key: str | None = field(default=None, repr=False)
    model: str = "deepseek-v4-flash"
    timeout: float = DEFAULT_SEARCH_TIMEOUT["deepseek"]
    max_retries: int = 1
    retry_delay: float = 1.0
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="deepseek")
    )

    def __post_init__(self) -> None:
        if deepseek_search is None or _resolve_api_key is None:
            raise RuntimeError(
                "deepseek_search 包未安装，无法使用 deepseek 搜索后端。"
                "请切换到 wikipedia 或 duckduckgo 后端。"
            )
        # 提前检查登录状态，避免 rollout 开始后才发现所有搜索都无法鉴权。
        self.api_key = _resolve_api_key(self.api_key)

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "DeepSeekSearchClient":
        """优先读取项目 .env，也兼容 deepseek-search login 保存的密钥。"""
        if env_path and load_dotenv is not None:
            load_dotenv(env_path)
        return cls(**kwargs)

    def search(self, query: str) -> SearchResult:
        """执行一次 Evidence 搜索；超时、429 和 5xx 最多有限重试。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            try:
                response = deepseek_search(
                    query,
                    api_key=self.api_key,
                    model=self.model,
                    timeout=self.timeout,
                    mode="evidence",
                )
                latency = time.perf_counter() - started
                items = parse_evidence(response.evidence)
                usage = response.usage
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += latency
                    self.stats.web_search_requests += response.total_search_requests
                    self.stats.result_count += response.result_count
                    self.stats.input_tokens += int(usage.get("input_tokens", 0))
                    self.stats.cache_read_input_tokens += int(
                        usage.get("cache_read_input_tokens", 0)
                    )
                    self.stats.output_tokens += int(usage.get("output_tokens", 0))
                return SearchResult(ok=True, items=items, latency=latency, status=200)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                retryable = status == 429 or status >= 500
                with self.stats._lock:
                    if status == 429 and not saw_rate_limit:
                        self.stats.rate_limits += 1
                        saw_rate_limit = True
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {status}", status)
            except httpx.TimeoutException:
                with self.stats._lock:
                    if not saw_timeout:
                        self.stats.timeouts += 1
                        saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except httpx.HTTPError as error:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, type(error).__name__)
            except RuntimeError as error:
                return self._error_result(started, str(error))

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class WikipediaSearchClient:
    """使用 Wikimedia Action API 搜索英文 Wikipedia 并返回页面正文片段。"""

    timeout: float = DEFAULT_SEARCH_TIMEOUT["wikipedia"]
    max_retries: int = 2
    retry_delay: float = 1.0
    min_request_interval: float = 0.31
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="wikipedia")
    )
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _next_request_time: float = field(default=0.0, init=False, repr=False)

    def search(self, query: str) -> SearchResult:
        """搜索 Top 3 页面；所有并发调用共享约 200 RPM 的启动限速。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        attempt = 0
        while True:
            self._wait_for_rate_slot()
            try:
                result = self._request(query, started)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    with self.stats._lock:
                        self.stats.rate_limits += 1
                    saw_rate_limit = True
                retryable = error.code == 429 or error.code >= 500
                if retryable and attempt < self.max_retries:
                    retry_after = error.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else self.retry_delay * (2**attempt)
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _wait_for_rate_slot(self) -> None:
        """序列化请求启动时间，避免超过 Wikimedia 的识别客户端分钟限额。"""
        with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_time = now + self.min_request_interval

    def _request(self, query: str, started: float) -> SearchResult:
        """一次请求同时执行全文搜索并取得前三个页面的纯文本摘要。"""
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "format": "json",
                "formatversion": 2,
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 3,
                "prop": "extracts|info",
                "explaintext": 1,
                "exintro": 1,
                "exchars": 1200,
                "inprop": "url",
                "redirects": 1,
                "utf8": 1,
            }
        )
        request = urllib.request.Request(
            f"{WIKIPEDIA_SEARCH_ENDPOINT}?{params}",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": WIKIPEDIA_USER_AGENT,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            payload = json.loads(body.decode("utf-8"))
            pages = payload.get("query", {}).get("pages", [])
            if not isinstance(pages, list):
                raise TypeError("Wikipedia 搜索响应 pages 不是列表")
            if any(not isinstance(page, dict) for page in pages):
                raise TypeError("Wikipedia 搜索响应 page 不是对象")
            ordered_pages = sorted(
                pages,
                key=lambda page: int(page.get("index", 1_000_000)),
            )
            items = [
                SearchItem(
                    title=str(page.get("title") or "Untitled").strip(),
                    content=str(page.get("extract") or "").strip(),
                    source="Wikipedia",
                    url=str(page.get("fullurl") or "").strip(),
                )
                for page in ordered_pages
                if str(page.get("extract") or "").strip()
            ]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换成 rollout 可观察、不中断训练的搜索结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class ZhihuSearchClient:
    """轮转使用多组凭证，并通过有限重试执行知乎搜索。"""

    access_secrets: str | list[str]
    timeout: float = DEFAULT_SEARCH_TIMEOUT["zhihu"]
    max_retries: int = 2
    retry_delay: float = 1.0
    stats: SearchStats = field(default_factory=lambda: SearchStats(backend="zhihu"))
    _next_secret_index: int = field(default=0, init=False, repr=False)
    _rate_limited_secret_indices: set[int] = field(
        default_factory=set, init=False, repr=False
    )
    _credential_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """清洗、去重凭证，并兼容直接传入单个字符串。"""
        raw_secrets = (
            [self.access_secrets]
            if isinstance(self.access_secrets, str)
            else self.access_secrets
        )
        secrets = list(
            dict.fromkeys(secret.strip() for secret in raw_secrets if secret.strip())
        )
        if not secrets:
            raise ValueError("至少需要一个知乎搜索 key")
        self.access_secrets = secrets

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "ZhihuSearchClient":
        """从逗号或换行分隔的环境变量读取一组搜索凭证。"""
        if env_path and load_dotenv is not None:
            load_dotenv(env_path)
        raw_secrets = (
            os.getenv("ZHIHU_SEARCH_KEYS")
            or os.getenv("ZHIHU_SEARCH_KEY")
            or os.getenv("ZHIHU_ACCESS_SECRET")
        )
        if not raw_secrets:
            raise ValueError(
                "请设置 ZHIHU_SEARCH_KEYS、ZHIHU_SEARCH_KEY 或 ZHIHU_ACCESS_SECRET"
            )
        secrets = [
            item.strip() for item in re.split(r"[,\n]", raw_secrets) if item.strip()
        ]
        return cls(access_secrets=secrets, **kwargs)

    def search(self, query: str) -> SearchResult:
        """轮转 key 搜索一个 query；429 切换 key，超时和 5xx 有限重试。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        saw_rate_limit = False
        credential = self._next_credential()
        if credential is None:
            return self._error_result(
                started, "all search keys are rate limited", 429
            )
        secret_index, access_secret = credential
        attempt = 0
        while True:
            try:
                result = self._request(query, started, access_secret)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                if error.code == 429 and not saw_rate_limit:
                    with self.stats._lock:
                        self.stats.rate_limits += 1
                    saw_rate_limit = True
                if error.code == 429:
                    self._disable_credential(secret_index)
                    credential = self._next_credential()
                    if credential is None:
                        return self._error_result(
                            started, "all search keys are rate limited", error.code
                        )
                    secret_index, access_secret = credential
                    with self.stats._lock:
                        self.stats.credential_failovers += 1
                    continue
                if error.code >= 500 and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                return self._error_result(started, type(error).__name__)

    def _next_credential(self) -> tuple[int, str] | None:
        """按 round-robin 顺序取下一组尚未被 429 停用的凭证。"""
        secrets = cast(list[str], self.access_secrets)
        with self._credential_lock:
            for _ in range(len(secrets)):
                index = self._next_secret_index
                self._next_secret_index = (self._next_secret_index + 1) % len(secrets)
                if index not in self._rate_limited_secret_indices:
                    return index, secrets[index]
        return None

    def _disable_credential(self, index: int) -> None:
        """把返回 429 的 key 标记为本次运行不可再用。"""
        with self._credential_lock:
            self._rate_limited_secret_indices.add(index)

    def _request(
        self, query: str, started: float, access_secret: str
    ) -> SearchResult:
        """发出一次知乎 API 请求并解析真实响应结构。"""
        params = urllib.parse.urlencode(
            {"Query": query, "Count": 3, "SearchDB": "all"}
        )
        request = urllib.request.Request(
            f"{SEARCH_ENDPOINT}?{params}",
            headers={
                "Authorization": f"Bearer {access_secret}",
                "X-Request-Timestamp": str(int(time.time())),
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            items = [self._parse_item(item) for item in payload["Data"]["Items"]]
            return SearchResult(
                ok=True,
                items=items,
                latency=time.perf_counter() - started,
                status=response.status,
            )

    def _parse_item(self, item: dict[str, Any]) -> SearchItem:
        """从一条 API 结果中保留标题、摘要、来源和链接。"""
        source_parts = [str(item.get("ContentType") or "Zhihu")]
        if item.get("AuthorName"):
            source_parts.append(str(item["AuthorName"]))
        return SearchItem(
            title=str(item.get("Title") or "Untitled").strip(),
            content=str(item.get("ContentText") or "").strip()[:1200],
            url=str(item.get("Url") or "").strip(),
            source=" / ".join(source_parts),
        )

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        """把请求异常转换为不会泄露密钥的工具结果。"""
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class BingSearchClient:
    """使用 cn.bing.com HTML 搜索，无需 API key，国内可用，支持中英文。"""

    timeout: float = DEFAULT_SEARCH_TIMEOUT["bing"]
    max_retries: int = 2
    retry_delay: float = 1.0
    max_results: int = 3
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="bing")
    )
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _next_request_time: float = field(default=0.0, init=False, repr=False)
    min_request_interval: float = 0.35  # ~170 RPM，Bing 不会限速

    def search(self, query: str) -> SearchResult:
        """执行一次 Bing 搜索，返回前 max_results 条结果。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        saw_timeout = False
        attempt = 0
        while True:
            self._wait_for_rate_slot()
            try:
                result = self._request(query, started)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except urllib.error.HTTPError as error:
                retryable = error.code == 429 or error.code >= 500
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, f"HTTP {error.code}", error.code)
            except (TimeoutError, socket.timeout):
                if not saw_timeout:
                    with self.stats._lock:
                        self.stats.timeouts += 1
                    saw_timeout = True
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except urllib.error.URLError as error:
                if isinstance(error.reason, (TimeoutError, socket.timeout)):
                    if not saw_timeout:
                        with self.stats._lock:
                            self.stats.timeouts += 1
                        saw_timeout = True
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay * (2**attempt))
                        attempt += 1
                        continue
                    return self._error_result(started, "request timeout")
                return self._error_result(started, type(error).__name__)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                return self._error_result(started, type(error).__name__)

    def _wait_for_rate_slot(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_time = now + self.min_request_interval

    def _request(self, query: str, started: float) -> SearchResult:
        """通过 cn.bing.com 搜索并解析 HTML 结果。"""
        params = urllib.parse.urlencode({
            "q": query,
            "setlang": "en",
            "mkt": "en-US",
            "cc": "US",
        })
        url = f"https://cn.bing.com/search?{params}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
                "Cookie": "SRCHHPGUSR=ADLT=MODERATE; _EDGE_S=mkt=en-us",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        items = self._parse_bing_html(html)
        return SearchResult(
            ok=True,
            items=items[: self.max_results],
            latency=time.perf_counter() - started,
            status=200,
        )

    @staticmethod
    def _parse_bing_html(html: str) -> list[SearchItem]:
        """从 Bing HTML 搜索结果页中提取标题、摘要和链接。"""
        results: list[SearchItem] = []
        # 每个结果在 <li class="b_algo"> ... </li> 中
        blocks = re.findall(
            r'<li\s+class="b_algo"[^>]*>(.*?)</li>', html, re.DOTALL
        )
        for block in blocks:
            # 提取链接和标题：<h2><a ... href="URL">TITLE</a></h2>
            link_match = re.search(
                r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>',
                block,
                re.DOTALL,
            )
            if not link_match:
                # 降级：<a target="_blank" ... href="URL">TITLE</a>
                link_match = re.search(
                    r'<a[^>]+target="_blank"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    block,
                    re.DOTALL,
                )
            if not link_match:
                continue
            href = link_match.group(1)
            title = re.sub(r"<[^>]+>", "", link_match.group(2)).strip()
            # 去除标题中混入的 URL 片段（如 "totalenergies.comhttps://totalenergies.com"）
            title = re.sub(r"^https?://\S+\s*", "", title).strip()
            title = re.sub(r"\s*https?://\S+", "", title).strip()
            # 提取摘要：<p class="b_lineclamp2" ...>TEXT</p>
            snippet = ""
            snippet_match = re.search(
                r'<p[^>]*class="b_lineclamp2"[^>]*>(.*?)</p>',
                block,
                re.DOTALL,
            )
            if snippet_match:
                snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip()
            # 提取来源：<cite>URL</cite>
            source = ""
            cite_match = re.search(r"<cite>(.*?)</cite>", block)
            if cite_match:
                source = re.sub(r"<[^>]+>", "", cite_match.group(1)).strip()
            if title:
                results.append(SearchItem(
                    title=title[:120],
                    content=snippet,
                    source=source or "Bing",
                    url=href,
                ))
        return results

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


@dataclass
class MiMoSearchClient:
    """使用小米 MiMo API 作为搜索后端。

    将搜索查询发送给 MiMo LLM，让它以搜索引擎的风格返回相关事实信息。
    适用于无可用 web search API 的场景（如 DeepSeek API 不可用、国内网络限制）。
    需要 API key（环境变量 MIMO_API_KEY 或直接传入）。
    """

    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.xiaomimimo.com/v1"
    model: str = "mimo-v2.5"
    timeout: float = DEFAULT_SEARCH_TIMEOUT["mimo"]
    max_retries: int = 2
    retry_delay: float = 1.0
    stats: SearchStats = field(
        default_factory=lambda: SearchStats(backend="mimo")
    )
    _rate_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _next_request_time: float = field(default=0.0, init=False, repr=False)
    min_request_interval: float = 0.1
    max_concurrency: int = 16
    _semaphore: threading.Semaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, '_semaphore', threading.Semaphore(self.max_concurrency))
        if not self.api_key:
            self.api_key = os.environ.get("MIMO_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "MiMo 搜索后端需要 API key。"
                "请设置 MIMO_API_KEY 环境变量或传入 api_key 参数。"
            )

    @classmethod
    def from_env(
        cls, env_path: str | Path | None = None, **kwargs: Any
    ) -> "MiMoSearchClient":
        if env_path and load_dotenv is not None:
            load_dotenv(env_path)
        return cls(**kwargs)

    def search(self, query: str) -> SearchResult:
        """将查询发送给 MiMo LLM，以搜索结果格式返回。"""
        started = time.perf_counter()
        with self.stats._lock:
            self.stats.requests += 1

        # 并发控制：最多 max_concurrency 个线程同时调 MiMo API
        self._semaphore.acquire()
        try:
            return self._search_inner(query, started)
        finally:
            self._semaphore.release()

    def _search_inner(self, query: str, started: float) -> SearchResult:
        attempt = 0
        while True:
            self._wait_for_rate_slot()
            try:
                result = self._request(query, started)
                with self.stats._lock:
                    self.stats.successes += 1
                    self.stats.latency_total += result.latency
                    self.stats.result_count += len(result.items)
                return result
            except (urllib.error.HTTPError, urllib.error.URLError) as error:
                status = None
                if isinstance(error, urllib.error.HTTPError):
                    status = error.code
                    retryable = status == 429 or status >= 500
                else:
                    retryable = True
                if retryable and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(
                    started,
                    f"HTTP {status}" if status else type(error).__name__,
                    status,
                )
            except (TimeoutError, socket.timeout):
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    attempt += 1
                    continue
                return self._error_result(started, "request timeout")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                return self._error_result(started, type(error).__name__)

    def _wait_for_rate_slot(self) -> None:
        with self._rate_lock:
            now = time.monotonic()
            delay = self._next_request_time - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request_time = now + self.min_request_interval

    def _request(self, query: str, started: float) -> SearchResult:
        """调用 MiMo chat API，以搜索风格回答查询。"""
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a search engine. Answer the query concisely "
                        "with factual information. Include key facts, numbers, "
                        "and names. Keep response under 200 words."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 256,
            "temperature": 0.3,
        }).encode("utf-8")

        url = f"{self.base_url}/chat/completions"
        request = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))

        choices = data.get("choices", [])
        if not choices:
            return SearchResult(
                ok=True, items=[], latency=time.perf_counter() - started, status=200
            )
        content = choices[0].get("message", {}).get("content", "").strip()
        if not content:
            return SearchResult(
                ok=True, items=[], latency=time.perf_counter() - started, status=200
            )

        usage = data.get("usage") or {}
        with self.stats._lock:
            self.stats.input_tokens += int(usage.get("prompt_tokens", 0))
            self.stats.output_tokens += int(usage.get("completion_tokens", 0))

        # 直接将 MiMo 回复作为单条搜索结果（不依赖格式解析）
        items = [SearchItem(title=query, content=content)]
        latency = time.perf_counter() - started
        return SearchResult(ok=True, items=items, latency=latency, status=200)

    def _error_result(
        self,
        started: float,
        message: str,
        status: int | None = None,
    ) -> SearchResult:
        latency = time.perf_counter() - started
        with self.stats._lock:
            self.stats.errors += 1
            self.stats.latency_total += latency
        return SearchResult(False, [], latency, status=status, error=message)


SearchClient = DeepSeekSearchClient | WikipediaSearchClient | ZhihuSearchClient | BingSearchClient | MiMoSearchClient


def resolve_search_concurrency(backend: str, value: int | None) -> int:
    """返回用户设置或当前后端的默认搜索并发。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    concurrency = DEFAULT_SEARCH_CONCURRENCY[backend] if value is None else value
    if concurrency < 1:
        raise ValueError("search_concurrency 必须大于等于 1")
    return concurrency


def resolve_search_timeout(backend: str, value: float | None) -> float:
    """返回用户设置或当前后端的默认请求超时。"""
    if backend not in SEARCH_BACKENDS:
        raise ValueError(f"不支持的搜索后端: {backend}")
    timeout = DEFAULT_SEARCH_TIMEOUT[backend] if value is None else value
    if timeout <= 0:
        raise ValueError("search_timeout 必须大于 0")
    return timeout


def create_search_client(
    backend: str,
    env_path: str | Path | None = None,
    *,
    model: str = "deepseek-v4-flash",
    timeout: float | None = None,
) -> SearchClient:
    """按名称创建搜索后端；四个后端共用同一套 rollout 接口。"""
    resolved_timeout = resolve_search_timeout(backend, timeout)
    if backend == "deepseek":
        return DeepSeekSearchClient.from_env(
            env_path,
            model=model,
            timeout=resolved_timeout,
        )
    if backend == "wikipedia":
        return WikipediaSearchClient(timeout=resolved_timeout)
    if backend == "zhihu":
        return ZhihuSearchClient.from_env(
            env_path,
            timeout=resolved_timeout,
        )
    if backend == "bing":
        return BingSearchClient(timeout=resolved_timeout)
    if backend == "mimo":
        return MiMoSearchClient.from_env(env_path, timeout=resolved_timeout)
    raise ValueError(f"不支持的搜索后端: {backend}")


def parse_evidence(evidence: str | None) -> list[SearchItem]:
    """把 Evidence 模式的编号文本拆成可独立截断的证据条目。"""
    if not evidence:
        return []
    # EVIDENCE_PATTERN 有 3 个 group: (编号, Source标题, Evidence内容)
    raw = EVIDENCE_PATTERN.findall(evidence.strip())
    items = [
        SearchItem(title=title.strip(), content=content.strip())
        for _num, title, content in raw
        if title.strip() and content.strip()
    ]
    if items:
        return items
    # 降级：尝试简单解析 [N] Title: ... Content: ... 格式
    fallback = re.findall(
        r'\[\d+\]\s*(?:Title:\s*)?(.*?)\s*(?:Content|Evidence):\s*(.*?)(?=\n\[|\Z)',
        evidence.strip(),
        re.DOTALL,
    )
    items = [
        SearchItem(title=title.strip(), content=content.strip())
        for title, content in fallback
        if title.strip() and content.strip()
    ]
    if items:
        return items
    return [SearchItem(title="Search Result", content=evidence.strip())]


def format_item(item: SearchItem, index: int) -> str:
    """按后端提供的信息格式化一条完整工具证据。"""
    if item.source is None and item.url is None:
        return f"[{index}] Source: {item.title}\n    Evidence: {item.content}"
    lines = [
        f"[{index}] Title: {item.title}",
        f"    Content: {item.content}",
    ]
    if item.source:
        lines.append(f"    Source: {item.source}")
    if item.url:
        lines.append(f"    URL: {item.url}")
    return "\n".join(lines)
