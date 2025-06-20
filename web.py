from httpx import AsyncClient


def get_img_name(img_url: str) -> str:
    return img_url.split("/")[-1].split(".")[0]


async def extract_web_page_content(api_key: str, url: str) -> str | None:
    """
    Extract web page content from URLs using Tavily Extract API.
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with AsyncClient() as client:
        response = await client.post(
            url="https://api.tavily.com/extract",
            headers=headers,
            json={"urls": [url], "include_images": True},
            timeout=30,
        )

    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        return None

    content = results[0]["raw_content"]
    images = results[0]["images"]
    if not images:
        return content

    images_list_md = "\n".join(f"- ![{get_img_name(image)}]({image})" for image in images)
    content += f"\n\n### Images: {images_list_md}"

    return content


async def google_search(api_key: str, search_engine_id: str, query: str, count) -> list | str:
    """
    Search using Google's Programmable Search Engine API and return the results as a list objects.
    Handles pagination for counts greater than 10.
    """

    url = "https://www.googleapis.com/customsearch/v1"
    headers = {"Content-Type": "application/json"}
    all_results = []
    start_index = 1  # Google PSE start parameter is 1-based

    while count > 0:
        num_results_this_page = min(count, 10)  # Google PSE max results per page is 10
        params = {
            "cx": search_engine_id,
            "q": query,
            "key": api_key,
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


async def brave_search(api_key: str, query: str, count: int) -> list | str:
    """
    Search using Brave's Search API and return the results as a list of objects.
    """

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": count,
        "result_filter": "web",
    }

    async with AsyncClient() as client:
        response = await client.get(url, headers=headers, params=params)

    response.raise_for_status()
    json_response = response.json()

    results = json_response.get("web", {}).get("results", [])

    return [{
        "link": result["url"],
        "title": result["title"],
        "snippet": result["description"]
    } for result in results]
