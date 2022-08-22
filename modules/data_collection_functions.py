# IEX Cloud Library
from iexfinance.stocks import get_historical_data

# Yahoo Finance Library
import yfinance as yf

import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()


def get_historical_data_from_yf(symbols, period):
    data = yf.download(list(symbols), period=period)
    return data


def get_retrieved_symbols(data):
    columns = data.columns
    retrieved = set()
    for column in columns:
        retrieved.add(column[0])
    return retrieved


def get_unretrieved_symbols(symbols, retrieved_symbols):
    unretrieved = symbols.difference(retrieved_symbols)
    return unretrieved


def export_to_csv(data, path, tickers):
    for ticker in tickers:
        df = data[ticker]
        df.to_csv(path + ticker + '.csv')
    return


def get_tasks(session, symbols, api_key, url):
    tasks = []
    for ticker in symbols:
        api_url = url.format(ticker, api_key)
        coroutine = session.get(api_url, ssl=False)
        tasks.append(asyncio.create_task(coroutine))
    return tasks


async def get_symbols(symbols, iex_api_key, url):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = get_tasks(session, symbols, iex_api_key, url)
        responses = await asyncio.gather(*tasks)
        for response in responses:
            results.append(await response.json())
    return results


def get_historical_data_from_iexcloud(symbols, start, end, iex_api_key):

    companies = dict()
    for symbol in symbols:
        data = get_historical_data(symbol, start, end, output_format='pandas', token=iex_api_key)
        companies[symbol] = data
    return companies
