"""
sentiment.py
============
This module performs sentiment analysis on financial news headlines
using the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool.

VADER is specifically tuned for social media / short-text sentiment and
works well on news headlines without any training.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


# Create a single analyzer instance (reused across calls)
analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a single piece of text.

    Parameters
    ----------
    text : str
        The text to analyze (e.g., a news headline).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'positive'  : Positive sentiment score (0 to 1)
        - 'negative'  : Negative sentiment score (0 to 1)
        - 'neutral'   : Neutral sentiment score (0 to 1)
        - 'compound'  : Overall sentiment (-1 = most negative, +1 = most positive)
    """
    scores = analyzer.polarity_scores(text)
    return {
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "compound": scores["compound"],
    }


def analyze_headlines(headlines: list) -> pd.DataFrame:
    """
    Analyze sentiment for a list of headlines and return results as a DataFrame.

    Parameters
    ----------
    headlines : list of str
        List of news headlines to analyze.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Headline, Positive, Negative, Neutral, Compound.
    """
    results = []
    for headline in headlines:
        scores = analyze_sentiment(headline)
        results.append({
            "Headline": headline,
            "Positive": scores["positive"],
            "Negative": scores["negative"],
            "Neutral": scores["neutral"],
            "Compound": scores["compound"],
        })
    return pd.DataFrame(results)


def get_average_sentiment(headlines: list) -> float:
    """
    Get the average compound sentiment score for a list of headlines.

    Parameters
    ----------
    headlines : list of str
        List of news headlines.

    Returns
    -------
    float
        Average compound score (-1 to +1). Returns 0 if the list is empty.
    """
    if not headlines:
        return 0.0
    total = sum(analyze_sentiment(h)["compound"] for h in headlines)
    return round(total / len(headlines), 4)


def get_sample_headlines() -> list:
    """
    Return a set of realistic sample financial news headlines.
    These are used when no live news API is available.

    Returns
    -------
    list of str
        List of sample headlines covering a mix of positive, negative, and neutral news.
    """
    return [
        "Stock market rallies to record highs amid strong earnings reports",
        "Federal Reserve signals potential interest rate cuts in upcoming meeting",
        "Tech sector faces massive sell-off as inflation fears grow",
        "Oil prices surge as OPEC announces production cuts",
        "Major bank reports unexpected quarterly losses, shares plummet",
        "Cryptocurrency market crashes as regulatory crackdown intensifies",
        "Manufacturing sector shows signs of recovery with positive PMI data",
        "Trade war escalation threatens global economic stability",
        "Consumer confidence index drops to lowest level in two years",
        "Goldman Sachs warns of potential market correction in 2024",
        "Unemployment rate falls to historic low, economy adds 300K jobs",
        "Hedge funds increase short positions signaling bearish outlook",
        "Real estate market cooling as mortgage rates hit 7 percent",
        "AI stocks soar as companies report record revenue growth",
        "Global supply chain disruptions continue to impact markets",
        "Investors flee to safe haven assets amid geopolitical tensions",
        "Retail sales exceed expectations boosting market sentiment",
        "Bond yields invert again raising recession probability",
        "Central banks worldwide coordinate emergency intervention",
        "Startup layoffs accelerate as venture capital funding dries up",
        # Extra 5 diverse samples added based on request
        "Catastrophic failure in major financial institution triggers widespread panic",
        "Historic bull run continues as S&P 500 breaks all-time records for 10th consecutive day",
        "Unexpected regulatory approval sends biotech stocks soaring by 300 percent",
        "Global cyberattack cripples trading platforms, investors fear massive losses",
        "Revolutionary energy breakthrough promises totally free electricity, markets stunned",
    ]
