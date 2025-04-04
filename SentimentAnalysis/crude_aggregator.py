import pandas as pd
import datetime
from gdeltdoc import GdeltDoc, Filters
import os
from tqdm import tqdm
import time

# Import the sentiment analysis function
# Assuming this function is defined in another file like sentiment_model.py
from crudebert import predict_scores

def get_crude_oil_titles(date_str):
    """
    Query GDELT for news titles related to crude oil on a specific date

    Args:
        date_str (str): Date in format 'YYYY-MM-DD'

    Returns:
        list: List of news titles related to crude oil
    """
    gdelt = GdeltDoc()

    # Parse date and create date range (single day)
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    next_day = (date_obj + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    # Create filters for crude oil related news
    f = Filters(
        keyword=["crude oil", "oil price", "petroleum", "oil futures"],  #, "WTI", "Brent"],
        # keyword="crude oil",
        start_date=date_str,
        end_date=next_day,
        num_records=250  # Max number of records allowed in one query,
        # country="US",  # Uncomment if you want to filter by country
    )

    # TODO: Adaptively constrain and loop filters such that all articles are fetched (not just the first 250)
    try:
        # Query GDELT
        articles = gdelt.article_search(f)
        titles = articles['title'].tolist()
        return titles
    except Exception as e:
        print(f"Error querying GDELT for {date_str}: {str(e)}")
        return []

def main():
    # Read crude oil data
    input_path = '../Local_Data/futures_full_30min_contin_UNadj_11assu1/CL_full_30min_continuous_UNadjusted.csv'
    output_path = '../Local_Data/crude_sentiments.csv'

    print(f"Reading crude oil data from {input_path}")
    crude_data = pd.read_csv(input_path)

    # Convert date column to datetime
    crude_data['date'] = pd.to_datetime(crude_data['date'])

    # Filter for 2015 to 2024
    filtered_data = crude_data[crude_data['date'].dt.year <= 2024]
    filtered_data = filtered_data[filtered_data['date'].dt.year >= 2015].copy()

    # Get unique dates
    unique_dates = filtered_data['date'].dt.strftime('%Y-%m-%d').unique()
    print(f"Found {len(unique_dates)} unique dates")

    # Create list to store sentiment results
    results = []

    # Process each date
    time_limit = 5.1
    last_start = time.time() - time_limit
    for date_str in tqdm(unique_dates):
        # Sleep to avoid hitting API rate limits
        curr_time = time.time()
        wait_time = time_limit - (curr_time - last_start)
        if wait_time > 0:
            time.sleep(wait_time)
            last_start = time.time()
        else:
            last_start = curr_time

        # Get news titles for this date
        titles = get_crude_oil_titles(date_str)

        if titles:
            # Calculate sentiment scores for all titles in batch
            sentiment_scores = predict_scores(titles)

            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            results.append({
                'date': date_str,
                'avg_sentiment': avg_sentiment,
                'num_articles': len(titles)
            })

            print(f"{date_str}: Found {len(titles)} articles, avg sentiment: {avg_sentiment:.4f}")
        else:
            # No articles found
            results.append({
                'date': date_str,
                'avg_sentiment': None,
                'num_articles': 0
            })

            print(f"{date_str}: No articles found")

        # Add delay to avoid hitting API rate limits
        time.sleep(1)

    # Create final DataFrame
    sentiment_df = pd.DataFrame(results)

    # Save to CSV
    sentiment_df.to_csv(output_path, index=False)
    print(f"Saved sentiment results to {output_path}")

if __name__ == "__main__":
    main()
