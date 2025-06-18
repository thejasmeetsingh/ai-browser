import os
from httpx import AsyncClient


async def extract_web_page_content(url: str) -> list:
    """
    Extract web page content from URLs using Tavily Extract API.
    """

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not configured in env")

    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json"
    }

    async with AsyncClient() as client:
        response = await client.post(
            url="https://api.tavily.com/extract",
            headers=headers,
            json={"urls": [url], "format": "text"},
            timeout=30,
        )

    response.raise_for_status()
    results = response.json().get("results", [])

    return results[0]['raw_content'] if results else ""


async def google_search(query: str, count: int = 10) -> list:
    """
    Search using Google's Programmable Search Engine API and return the results as a list objects.
    Handles pagination for counts greater than 10.
    """

    google_search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    google_pse_api_key = os.getenv("GOOGLE_PSE_API_KEY")

    if not google_search_engine_id or not google_pse_api_key:
        raise ValueError("Google PSE credentials are not configured in env")

    url = "https://www.googleapis.com/customsearch/v1"
    headers = {"Content-Type": "application/json"}
    all_results = []
    start_index = 1  # Google PSE start parameter is 1-based

    while count > 0:
        num_results_this_page = min(count, 10)  # Google PSE max results per page is 10
        params = {
            "cx": google_search_engine_id,
            "q": query,
            "key": google_pse_api_key,
            "num": num_results_this_page,
            "start": start_index,
        }

        async with AsyncClient() as client:
            response = await client.get(url=url, params=params, headers=headers)

        response.raise_for_status()
        json_response = response.json()

        results = json_response.get("items", [])

        if results:  # check if results are returned. If not, no more pages to fetch.
            all_results.extend(results)
            count -= len(
                results
            )  # Decrement count by the number of results fetched in this page.
            start_index += 10  # Increment start index for the next page
        else:
            break  # No more results from Google PSE, break the loop

    return [{
        "link": result["link"],
        "title": result["title"],
        "snippet": result["snippet"]
    } for result in all_results]
