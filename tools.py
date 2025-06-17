import os
from httpx import AsyncClient


async def extract_web_page_content(url: str) -> list:
    headers = {
        "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",
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
    url = "https://www.googleapis.com/customsearch/v1"
    headers = {"Content-Type": "application/json"}
    all_results = []
    start_index = 1  # Google PSE start parameter is 1-based

    while count > 0:
        num_results_this_page = min(count, 10)  # Google PSE max results per page is 10
        params = {
            "cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
            "q": query,
            "key": os.getenv("GOOGLE_PSE_KEY"),
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
