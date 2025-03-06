import requests
import json
import sys


def get_news_sentiment(ticker=None, api_key=None, topics=None, time_from=None,
                       time_to=None, sort="LATEST", limit=50):
    """
    Query Alpha Vantage for market news & sentiment data with all available parameters.

    Parameters:
        ticker (str, optional): Stock/crypto/forex symbols (comma-separated). Example: 'IBM' or 'COIN,CRYPTO:BTC'
        api_key (str): Your Alpha Vantage API key
        topics (str, optional): News topics to filter (comma-separated). Available topics:
            blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets,
            economy_fiscal, economy_monetary, economy_macro, energy_transportation,
            finance, life_sciences, manufacturing, real_estate, retail_wholesale, technology
        time_from (str, optional): Start time in YYYYMMDDTHHMM format. Example: '20220410T0130'
        time_to (str, optional): End time in YYYYMMDDTHHMM format
        sort (str, optional): Sorting order - 'LATEST', 'EARLIEST', or 'RELEVANCE'. Default: 'LATEST'
        limit (int, optional): Maximum number of results (1-1000). Default: 50

    Returns:
        dict: The JSON response from Alpha Vantage as a Python dictionary
    """
    # Base URL with required function parameter
    url = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT'

    # Add optional parameters if provided
    if ticker:
        url += f'&tickers={ticker}'
    if topics:
        url += f'&topics={topics}'
    if time_from:
        url += f'&time_from={time_from}'
    if time_to:
        url += f'&time_to={time_to}'
    if sort:
        url += f'&sort={sort}'
    if limit:
        url += f'&limit={limit}'

    # Add API key
    url += f'&apikey={api_key}'

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error: HTTP {response.status_code}")

    data = response.json()
    if "Error Message" in data:
        raise Exception(f"API Error: {data['Error Message']}")

    return data


def main():
    # Use ticker provided as command-line argument or default to "CL"
    ticker = sys.argv[1] if len(sys.argv) > 1 else "CL"
    # Replace with your Alpha Vantage API key
    api_key = "08RPYK8BYLTKW10D"

    try:
        # Example with additional parameters
        sentiment_data = get_news_sentiment(
            ticker=ticker,
            api_key=api_key,
            time_from="20250301T0000",
            time_to="20250305T0000",
            limit=100,
            sort="RELEVANCE"
        )
        # Pretty-print the JSON data
        print(json.dumps(sentiment_data, indent=4))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()