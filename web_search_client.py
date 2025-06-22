import os
import asyncio
import hashlib
from dataclasses import dataclass
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Set

import httpx
from rich.console import Console
from httpx import AsyncClient, HTTPStatusError, RequestError, TimeoutException


console = Console()


@dataclass
class SearchResult:
    """
    Standardized search result data structure.
    
    Attributes:
        link: URL of the search result
        title: Title of the page
        snippet: Brief description/snippet
        source: Search provider that returned this result
        rank: Position in search results (optional)
    """
    link: str
    title: str
    snippet: str
    source: str
    rank: Optional[int] = None
    
    def __post_init__(self):
        """Validate and clean up the search result data."""
        self.link = self.link.strip() if self.link else ""
        self.title = self.title.strip() if self.title else "No Title"
        self.snippet = self.snippet.strip() if self.snippet else "No Description"
        
    @property
    def domain(self) -> str:
        """Extract domain from the URL."""
        try:
            return urlparse(self.link).netloc
        except Exception:
            return "unknown"
    
    @property
    def url_hash(self) -> str:
        """Generate a hash of the URL for deduplication."""
        return hashlib.md5(self.link.encode()).hexdigest()
    
    def to_dict(self) -> dict[str, str]:
        """Get dict representation of SearchResult"""
        return {
            "link": self.link,
            "title": self.title,
            "snippet": self.snippet,
            "source": self.source,
            "rank": self.rank
        }


class WebSearchError(Exception):
    """Custom exception for web search related errors."""
    pass


class ContentExtractionError(Exception):
    """Custom exception for content extraction related errors."""
    pass


class WebSearchClient:
    """
    Enhanced web search and content extraction client.
    
    Provides unified interface for multiple search providers and content extraction
    with built-in error handling, rate limiting, and result deduplication.
    
    Attributes:
        timeout: Default timeout for HTTP requests
        max_retries: Maximum number of retry attempts
        deduplicate: Whether to remove duplicate results
    """
    
    def __init__(
        self, 
        timeout: float = 30.0, 
        max_retries: int = 3,
        deduplicate: bool = True
    ):
        """
        Initialize the web search client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            deduplicate: Whether to remove duplicate search results
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.deduplicate = deduplicate
        
        # API configurations
        self.api_configs = self._load_api_configs()
        
        console.log(f"[blue]WebSearchClient initialized with timeout={timeout}s, retries={max_retries}[/blue]")

    def _load_api_configs(self) -> Dict[str, Dict[str, str]]:
        """Load API configurations from environment variables."""
        configs = {
            'tavily': {
                'api_key': os.getenv("TAVILY_API_KEY"),
                'base_url': "https://api.tavily.com"
            },
            'google': {
                'api_key': os.getenv("GOOGLE_API_KEY"),
                'search_engine_id': os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                'base_url': "https://www.googleapis.com/customsearch/v1"
            },
            'brave': {
                'api_key': os.getenv("BRAVE_API_KEY"),
                'base_url': "https://api.search.brave.com/res/v1/web/search"
            }
        }
        
        # Log which APIs are configured
        configured_apis = [name for name, config in configs.items() 
                          if config.get('api_key')]
        console.log(f"[blue]Configured APIs: {configured_apis}[/blue]")
        
        return configs

    async def _make_request_with_retry(
        self, 
        client: AsyncClient, 
        method: str, 
        url: str, 
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with retry logic.
        
        Args:
            client: HTTPX async client
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            HTTP response object
            
        Raises:
            WebSearchError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
                
            except TimeoutException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    console.log(f"[yellow]Request timeout, retrying in {wait_time}s (attempt {attempt + 1})[/yellow]")
                    await asyncio.sleep(wait_time)
                    continue
                    
            except HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    if attempt < self.max_retries:
                        wait_time = 5 * (attempt + 1)  # Longer wait for rate limits
                        console.log(f"[yellow]Rate limited, waiting {wait_time}s (attempt {attempt + 1})[/yellow]")
                        await asyncio.sleep(wait_time)
                        continue
                last_exception = e
                break  # Don't retry for other HTTP errors
                
            except RequestError as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    console.log(f"[yellow]Request error, retrying in {wait_time}s (attempt {attempt + 1})[/yellow]")
                    await asyncio.sleep(wait_time)
                    continue
        
        raise WebSearchError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")

    async def extract_content(self, url: str) -> str:
        """
        Extract web page content from URL using Tavily Extract API.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted raw content as string
            
        Raises:
            ContentExtractionError: If extraction fails or API not configured
        """
        tavily_config = self.api_configs.get('tavily', {})
        api_key = tavily_config.get('api_key')
        
        if not api_key:
            raise ContentExtractionError("Tavily API key not configured")
        
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {"urls": [url.strip()]}

        async with AsyncClient(timeout=self.timeout) as client:
            try:
                response = await self._make_request_with_retry(
                    client, 
                    "POST",
                    f"{tavily_config['base_url']}/extract",
                    headers=headers,
                    json=payload
                )
                
                results = response.json().get("results", [])
                
                if not results:
                    console.log(f"[yellow]No content extracted from {url}[/yellow]")
                    return ""
                
                content = results[0].get("raw_content", "")
                console.log(f"[blue]Successfully extracted {len(content)} characters from {url}[/blue]")
                return content
                
            except Exception as e:
                raise ContentExtractionError(f"Failed to extract content from {url}: {e}")

    async def google_search(self, query: str, count: int = 10) -> List[SearchResult]:
        """
        Search using Google's Programmable Search Engine API.
        
        Args:
            query: Search query string
            count: Number of results to return (handles pagination automatically)
            
        Returns:
            List of SearchResult objects
            
        Raises:
            WebSearchError: If search fails or API not configured
        """
        google_config = self.api_configs.get('google', {})
        api_key = google_config.get('api_key')
        search_engine_id = google_config.get('search_engine_id')

        if not api_key or not search_engine_id:
            console.log("[yellow]Google Search API not configured[/yellow]")
            return []

        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if count <= 0:
            raise ValueError("Count must be positive")

        headers = {"Content-Type": "application/json"}
        results = []
        start_index = 1  # Google CSE uses 1-based indexing
        remaining_count = count

        async with AsyncClient(timeout=self.timeout) as client:
            while remaining_count > 0:
                # Google CSE max results per page is 10
                page_size = min(remaining_count, 10)
                
                params = {
                    "cx": search_engine_id,
                    "q": query.strip(),
                    "key": api_key,
                    "num": page_size,
                    "start": start_index,
                }

                try:
                    response = await self._make_request_with_retry(
                        client, "GET", google_config['base_url'], 
                        params=params, headers=headers
                    )
                    
                    json_response = response.json()
                    items = json_response.get("items", [])

                    if not items:
                        console.log(f"[blue]No more results from Google for query: {query}[/blue]")
                        break

                    # Convert to SearchResult objects
                    for i, item in enumerate(items):
                        result = SearchResult(
                            link=item.get("link", ""),
                            title=item.get("title", ""),
                            snippet=item.get("snippet", ""),
                            source="google",
                            rank=start_index + i
                        )
                        results.append(result)

                    remaining_count -= len(items)
                    start_index += 10

                except Exception as e:
                    console.log(f"[red]Google search failed: {e}[/red]")
                    break

        console.log(f"[blue]Google search returned {len(results)} results for '{query}'[/blue]")
        return results

    async def brave_search(self, query: str, count: int = 10) -> List[SearchResult]:
        """
        Search using Brave's Search API.
        
        Args:
            query: Search query string
            count: Number of results to return
            
        Returns:
            List of SearchResult objects
            
        Raises:
            WebSearchError: If search fails or API not configured
        """
        brave_config = self.api_configs.get('brave', {})
        api_key = brave_config.get('api_key')

        if not api_key:
            console.log("[yellow]Brave Search API not configured[/yellow]")
            return []

        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if count <= 0:
            raise ValueError("Count must be positive")

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }

        params = {
            "q": query.strip(),
            "count": min(count, 50),  # Brave API max is 50
            "result_filter": "web",
        }

        async with AsyncClient(timeout=self.timeout) as client:
            try:
                response = await self._make_request_with_retry(
                    client, "GET", brave_config['base_url'],
                    headers=headers, params=params
                )
                
                json_response = response.json()
                web_results = json_response.get("web", {}).get("results", [])

                results = []
                for i, item in enumerate(web_results):
                    result = SearchResult(
                        link=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        source="brave",
                        rank=i + 1
                    )
                    results.append(result)

                console.log(f"[blue]Brave search returned {len(results)} results for '{query}'[/blue]")
                return results

            except Exception as e:
                console.log(f"[red]Brave search failed: {e}[/red]")
                return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of search results
        """
        if not self.deduplicate:
            return results
        
        seen_hashes: Set[str] = set()
        unique_results = []
        
        for result in results:
            url_hash = result.url_hash
            # Check if the URL is already see in dataset,
            # Also remove result which a YouTube URL since we are focusing on text based content only.
            if url_hash not in seen_hashes and "youtube" not in result.link.lower():
                seen_hashes.add(url_hash)
                unique_results.append(result)
        
        removed_count = len(results) - len(unique_results)
        if removed_count > 0:
            console.log(f"[blue]Removed {removed_count} duplicate results[/blue]")
        
        return unique_results

    async def search(
        self, 
        query: str, 
        count: int = 10,
        providers: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Unified search across multiple providers.
        
        Args:
            query: Search query string
            count: Number of results to return per provider
            providers: List of providers to use ('google', 'brave'). 
                      If None, uses all configured providers.
            
        Returns:
            Deduplicated and sorted list of SearchResult objects
            
        Raises:
            ValueError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string")

        if count <= 0:
            raise ValueError("Count must be positive")

        # Determine which providers to use
        if providers is None:
            providers = ['google', 'brave']
        
        # Validate providers
        valid_providers = {'google', 'brave'}
        providers = [p for p in providers if p in valid_providers]
        
        if not providers:
            raise ValueError(f"No valid providers specified. Valid options: {valid_providers}")

        # Execute searches concurrently
        search_tasks = []
        
        if 'google' in providers:
            search_tasks.append(self.google_search(query, count))
        
        if 'brave' in providers:
            search_tasks.append(self.brave_search(query, count))

        try:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except Exception as e:
            raise WebSearchError(f"Search execution failed: {e}")

        # Combine results
        all_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                console.log(f"[red]Search provider {providers[i]} failed: {result}[/red]")
                continue
            all_results.extend(result)

        # Deduplicate and sort
        unique_results = self._deduplicate_results(all_results)
        
        # Sort by source priority (Google first) and then by original rank
        def sort_key(result: SearchResult) -> tuple:
            source_priority = 0 if result.source == 'google' else 1
            return (source_priority, result.rank or 999)
        
        sorted_results = sorted(unique_results, key=sort_key)
        
        console.log(f"[blue]Combined search returned {len(sorted_results)} unique results for '{query}'[/blue]")
        return sorted_results

    def get_configured_providers(self) -> List[str]:
        """
        Get list of configured search providers.
        
        Returns:
            List of provider names that have valid API keys
        """
        configured = []
        
        if self.api_configs.get('google', {}).get('api_key') and \
           self.api_configs.get('google', {}).get('search_engine_id'):
            configured.append('google')
        
        if self.api_configs.get('brave', {}).get('api_key'):
            configured.append('brave')
        
        return configured
