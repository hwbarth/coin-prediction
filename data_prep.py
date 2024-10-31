#data preparation module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import talib
from technical_indicators import *
import argparse

def main(ticker, frequency):
    print(f"Ticker: {ticker}")
    print(f"Frequency: {frequency}")
    # You can add your code logic here, e.g., fetching data based on ticker and frequency

    data_path = 'data/Kraken_OHLCVT'


    # file_names = os.listdir(data_path)

    try:
        df = pd.read_csv(data_path + f"/{ticker}_{frequency}.csv", header=None)
    except: 
        raise
    # df["timestamp"] = pd.to_datetime(df[0], unit='s') 
    # X = df[[4, 'timestamp']]
    # X = df[[0, 4]]
    df = df.drop(columns=[6])


    #above march 9 for now
    df = df[df[0] >= 1710028800]


    x = df.to_numpy()

    # technical indicators
    bollingerbands = calculate_bollinger_bands(x)
    dema = calculate_dema(x)
    ema = calculate_ema(x)
    hilberttrendline = calculate_hilbert_trendline(x)
    kamam = calculate_kamam(x)
    midpoint = calculate_midpoint(x)
    midprice = calculate_midprice(x)
    sar = calculate_sar(x)
    sma = calculate_sma(x)
    tema = calculate_tema(x)
    wma = calculate_wma(x)
    adx = calculate_adx(x)
    adxr = calculate_adxr(x)
    apO = calculate_apO(x)
    aroonosc = calculate_aroonosc(x)
    bop = calculate_bop(x)
    cci = calculate_cci(x)
    cmo = calculate_cmo(x)
    dx = calculate_dx(x)
    macd = calculate_macd(x)
    minusdi = calculate_minus_di(x)
    minusdm = calculate_minus_dm(x)
    momentum = calculate_momentum(x)
    plusdi = calculate_plus_di(x)
    plusdm = calculate_plus_dm(x)
    ppo = calculate_ppo(x)
    roc = calculate_roc(x)
    rocp = calculate_rocp(x)
    rocr = calculate_rocr(x)
    rocr100 = calculate_rocr100(x)
    rsi = calculate_rsi(x)
    stochastic = calculate_stochastic(x)
    trix = calculate_trix(x)
    ultosc = calculate_ultosc(x)
    willr = calculate_willr(x)
    atr = calculate_atr(x)
    natr = calculate_natr(x)
    trange = calculate_trange(x)
    twocrows = calculate_two_crows(x)
    threeblackcrows = calculate_three_black_crows(x)
    threeinside = calculate_three_inside(x)
    threelinestrike = calculate_three_line_strike(x)
    threeoutside = calculate_three_outside(x)
    threestarsinsouth = calculate_three_stars_in_south(x)
    threeadvancingwhite = calculate_three_advancing_white(x)
    # abandonedbaby = calculate_abandoned_baby(x)
    advanceblock = calculate_advance_block(x)
    belthold = calculate_belt_hold(x)
    breakaway = calculate_breakaway(x)
    closingmarubozu = calculate_closing_marubozu(x)
    # concealingbabyswallow = calculate_concealing_baby_swallow(x)
    counterattack = calculate_counterattack(x)
    # darkcloudcover = calculate_dark_cloud_cover(x)
    # doji = calculate_doji(x)
    dojistar = calculate_doji_star(x)
    # dragonflydoji = calculate_dragonfly_doji(x)
    # engulfing = calculate_engulfing(x)
    eveningdojistar = calculate_evening_doji_star(x)
    eveningstar = calculate_evening_star(x)
    updowngapside = calculate_up_down_gap_side(x)
    gravestonedoji = calculate_gravestone_doji(x)
    hammer = calculate_hammer(x)
    hangingman = calculate_hanging_man(x)
    harami = calculate_harami(x)
    # haramicross = calculate_harami_cross(x)
    highwave = calculate_high_wave(x)
    hikkake = calculate_hikkake(x)
    modifiedhikkake = calculate_modified_hikkake(x)
    homingpigeon = calculate_homing_pigeon(x)
    identicalthreecrows = calculate_identical_three_crows(x)
    inneck = calculate_in_neck(x)
    # invertedhammer = calculate_inverted_hammer(x)
    # kicking = calculate_kicking(x)
    kickingbylength = calculate_kicking_by_length(x)
    ladderbottom = calculate_ladder_bottom(x)
    longleggeddoji = calculate_long_legged_doji(x)
    longline = calculate_long_line(x)
    marubozu = calculate_marubozu(x)
    matchinglow = calculate_matching_low(x)
    mathold = calculate_mat_hold(x)
    morningdojistar = calculate_morning_doji_star(x)
    morningstar = calculate_morning_star(x)
    onneck = calculate_on_neck(x)
    piercing = calculate_piercing(x)
    # rickshawman = calculate_rickshaw_man(x)
    risingfallingthreemethods = calculate_rising_falling_three_methods(x)
    separatinglines = calculate_separating_lines(x)
    shootingstar = calculate_shooting_star(x)
    shortline = calculate_short_line(x)
    spinningtop = calculate_spinning_top(x)
    stalledpattern = calculate_stalled_pattern(x)
    sticksandwich = calculate_stick_sandwich(x)
    takuri = calculate_takuri(x)
    tasukigap = calculate_tasuki_gap(x)
    thrusting = calculate_thrusting(x)
    # tristar = calculate_tristar(x)
    uniquethreeriver = calculate_unique_three_river(x)
    upsidegaptwocrows = calculate_upside_gap_two_crows(x)
    upsidedownsidegapthreemethods = calculate_upside_downside_gap_three_methods(x)
    hilberttransformdominantcycleperiod = calculate_hilbert_transform_dominant_cycle_period(x)
    hilberttransformdominantcyclephase = calculate_hilbert_transform_dominant_cycle_phase(x)
    hilberttransformtrendmode = calculate_hilbert_transform_trend_mode(x)


    # Create a dictionary to hold the dataframes
    dfs = {
        "bollingerbands": calculate_bollinger_bands(x),
        "dema": calculate_dema(x),
        "ema": calculate_ema(x),
        "hilberttrendline": calculate_hilbert_trendline(x),
        "kamam": calculate_kamam(x),
        "midpoint": calculate_midpoint(x),
        "midprice": calculate_midprice(x),
        "sar": calculate_sar(x),
        "sma": calculate_sma(x),
        "tema": calculate_tema(x),
        "wma": calculate_wma(x),
        "adx": calculate_adx(x),
        "adxr": calculate_adxr(x),
        "apo": calculate_apO(x),
        "aroonosc": calculate_aroonosc(x),
        "bop": calculate_bop(x),
        "cci": calculate_cci(x),
        "cmo": calculate_cmo(x),
        "dx": calculate_dx(x),
        "macd": calculate_macd(x),
        "minusdi": calculate_minus_di(x),
        "minusdm": calculate_minus_dm(x),
        "momentum": calculate_momentum(x),
        "plusdi": calculate_plus_di(x),
        "plusdm": calculate_plus_dm(x),
        "ppo": calculate_ppo(x),
        "roc": calculate_roc(x),
        "rocp": calculate_rocp(x),
        "rocr": calculate_rocr(x),
        "rocr100": calculate_rocr100(x),
        "rsi": calculate_rsi(x),
        "stochastic": calculate_stochastic(x),
        "trix": calculate_trix(x),
        "ultosc": calculate_ultosc(x),
        "willr": calculate_willr(x),
        "atr": calculate_atr(x),
        "natr": calculate_natr(x),
        "trange": calculate_trange(x),
        "twocrows": calculate_two_crows(x),
        "threeblackcrows": calculate_three_black_crows(x),
        "threeinside": calculate_three_inside(x),
        "threelinestrike": calculate_three_line_strike(x),
        "threeoutside": calculate_three_outside(x),
        "threestarsinsouth": calculate_three_stars_in_south(x),
        "threeadvancingwhite": calculate_three_advancing_white(x),
        "advanceblock": calculate_advance_block(x),
        "belthold": calculate_belt_hold(x),
        "breakaway": calculate_breakaway(x),
        "closingmarubozu": calculate_closing_marubozu(x),
        "counterattack": calculate_counterattack(x),
        "dojistar": calculate_doji_star(x),
        "eveningdojistar": calculate_evening_doji_star(x),
        "eveningstar": calculate_evening_star(x),
        "updowngapside": calculate_up_down_gap_side(x),
        "gravestonedoji": calculate_gravestone_doji(x),
        "hammer": calculate_hammer(x),
        "hangingman": calculate_hanging_man(x),
        "harami": calculate_harami(x),
        "kickingbylength": calculate_kicking_by_length(x),
        "ladderbottom": calculate_ladder_bottom(x),
        "longleggeddoji": calculate_long_legged_doji(x),
        "longline": calculate_long_line(x),
        "marubozu": calculate_marubozu(x),
        "matchinglow": calculate_matching_low(x),
        "mathold": calculate_mat_hold(x),
        "morningdojistar": calculate_morning_doji_star(x),
        "morningstar": calculate_morning_star(x),
        "onneck": calculate_on_neck(x),
        "piercing": calculate_piercing(x),
        "risingfallingthreemethods": calculate_rising_falling_three_methods(x),
        "separatinglines": calculate_separating_lines(x),
        "shootingstar": calculate_shooting_star(x),
        "shortline": calculate_short_line(x),
        "spinningtop": calculate_spinning_top(x),
        "stalledpattern": calculate_stalled_pattern(x),
        "sticksandwich": calculate_stick_sandwich(x),
        "takuri": calculate_takuri(x),
        "tasukigap": calculate_tasuki_gap(x),
        "thrusting": calculate_thrusting(x),
        "uniquethreeriver": calculate_unique_three_river(x),
        "upsidegaptwocrows": calculate_upside_gap_two_crows(x),
        "upsidedownsidegapthreemethods": calculate_upside_downside_gap_three_methods(x),
        "hilberttransformdominantcycleperiod": calculate_hilbert_transform_dominant_cycle_period(x),
        "hilberttransformdominantcyclephase": calculate_hilbert_transform_dominant_cycle_phase(x),
        "hilberttransformtrendmode": calculate_hilbert_transform_trend_mode(x)
    }

    # Start with the first DataFrame (assuming the first calculated indicator has data)
    merged_df = pd.DataFrame(dfs["bollingerbands"])

    # Merge each DataFrame on the timestamp
    for key, value in dfs.items():
        if key != "bollingerbands":  # Skip the first since it's already merged
            temp_df = pd.DataFrame(value)
            
            # Rename columns to avoid conflicts
            temp_df.columns = [f"{key}_{col}" if col != temp_df.columns[0] else col for col in temp_df.columns]
            
            # Merge on the timestamp
            merged_df = pd.merge(merged_df, temp_df, on=merged_df.columns[0], how='outer')

    # Optionally sort by timestamp
    merged_df.sort_values(by=merged_df.columns[0], inplace=True)

    # Reset index if needed
    merged_df.reset_index(drop=True, inplace=True)

    merged_df['timestamp'] = merged_df[0]
    prices = pd.DataFrame(x, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.merge(prices, merged_df, on='timestamp', how='inner', suffixes=('_price', '_indicator'))



    # df['returns'] = df['close'].pct_change() 
    # df['next_returns'] = df['returns'].shift(-1)
    # df = df.iloc[:-1]
    df = calculate_future_returns(df)

    file_name = f'data/silver_prices/{ticker}_{frequency}_silver.csv'
    try:
        df.to_csv(file_name, index=False)
        print('saved')
    except:
        print('fail')

def calculate_future_returns(df):
    returns = {}
    periods = [2, 4, 8, 16, 32, 64, 256]

    for period in periods:
        returns[period] = []

        for i in range(len(df.close)):
            current_price = df.iloc[i].close
            try:
                future_price = df.iloc[i + period].close
            except:
                future_return = None
                continue

            future_return = (future_price - current_price) / current_price
            returns[period].append(future_return)
    

    for period in periods:
        if len(returns[period]) != len(df):
            for i in range(len(df)):
                if i >= len(returns[period]):
                    returns[period].append(None)
                else:
                    continue
        df[f"return_{period}n"] = returns[period]



    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process cryptocurrency data.')
    parser.add_argument('ticker', type=str, help='The cryptocurrency ticker symbol (e.g., BTC, ETH)')
    parser.add_argument('frequency', type=int, help='The frequency of data points (e.g., 1 for daily, 7 for weekly)')

    args = parser.parse_args()
    main(args.ticker, args.frequency)
