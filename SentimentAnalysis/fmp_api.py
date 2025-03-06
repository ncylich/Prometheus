import requests
import json


def get_crude_sentiment(symbol="CL", from_date="2023-01-01", to_date="2023-03-01"):
    """
    Retrieves Crude Oil (CL) Commitment of Traders sentiment data
    from the Financial Modeling Prep API between two dates.
    """

    # Your FMP API key
    API_KEY = "lJXXqmQcNuEn5HRU8krQQ98iOLNlmh5M"

    # Base endpoint for COT analysis by symbol
    base_url = "https://financialmodelingprep.com/api/v4/commitment_of_traders_report_analysis"

    # Build request URL
    # If you prefer the full raw COT data, switch the endpoint to:
    #   base_url = "https://financialmodelingprep.com/api/v4/commitment_of_traders_report"
    #   and then use /CL instead of ?symbol=CL
    url = (
        f"{base_url}/{symbol}"
        f"?from={from_date}&to={to_date}"
        f"&apikey={API_KEY}"
    )

    print(f"Requesting data from: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Print raw data (as pretty JSON)
        print(json.dumps(data, indent=2))

        # Optional: parse the response for specific fields
        # Example: Net positions of "Managed Money"
        # The structure of the result depends on FMP’s returned JSON format
        for entry in data:
            date = entry.get("date")
            managed_money_long = entry.get("managedMoneyLong")
            managed_money_short = entry.get("managedMoneyShort")
            net_managed_money = None
            if managed_money_long is not None and managed_money_short is not None:
                net_managed_money = managed_money_long - managed_money_short

            print(f"Date: {date}, Managed Money Net: {net_managed_money}")

        return data
    else:
        print(f"Failed to retrieve data. HTTP {response.status_code} - {response.text}")
        return None


if __name__ == "__main__":
    # Example usage
    get_crude_sentiment(
        symbol="CL",  # Or whatever symbol your FMP plan uses for Crude Oil
        from_date="2023-07-01",  # Start date
        to_date="2023-09-01"  # End date
    )