from gdeltdoc import GdeltDoc, Filters  # https://github.com/alex9smith/gdelt-doc-api

# UTC time is used
# Get themese from http://data.gdeltproject.org/api/v2/guides/LOOKUP-GKGTHEMES.TXT

# Search query paramters
f = Filters(
    keyword = "crude oil",
    start_date = "2025-03-06",
    end_date = "2025-03-07",
    country = "US",
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)  # returns DF
articles = articles[['title', 'url']]
print(articles.columns)
print(articles.head())
print(articles.shape)
print()

# Get a timeline of the number of articles matching the filters
timeline = gd.timeline_search("timelinevol", f)
print(timeline.columns)
print(timeline.head())
print(timeline.shape)
